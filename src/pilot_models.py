"""06 — Model dan loop training (fenotipik, 2.5D, SFCN, DINOv2) dengan checkpoint per epoch + resume."""
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

import pilot_common as pc

LOG = logging.getLogger("pilot")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StopRequested(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def save_torch_atomic(obj, dst):
    dst = Path(dst)
    tmp = dst.with_name(f".{dst.name}.tmp")
    torch.save(obj, tmp)
    pc.sp._replace_retry(tmp, dst)


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, tr


# ==================================================================================================
# Fenotipik
# ==================================================================================================
def pheno_preprocessor(features):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, FunctionTransformer
    num = [c for c in features if c != "SEX"]
    tr = [("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                            ("sc", StandardScaler())]), num)]
    if "SEX" in features:   # ABIDE: 1 = laki-laki, 2 = perempuan -> biner
        tr.append(("sex", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("bin", FunctionTransformer(lambda x: (x == 2).astype(float)))]), ["SEX"]))
    return ColumnTransformer(tr)


class TabMLP(nn.Module):
    def __init__(self, d_in, hidden, dropout):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_torch_tabular(kind, hp, Xtr, ytr, Xva, yva, seed):
    """MLP / TabM dengan early stopping pada inner val (val loss). Cepat; tanpa checkpoint per epoch."""
    seed_all(seed)
    Xtr_t, Xva_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    ytr_t, yva_t = torch.tensor(ytr, dtype=torch.float32), torch.tensor(yva, dtype=torch.float32)
    if kind == "tabm":
        from tabm import TabM
        model = TabM.make(n_num_features=Xtr.shape[1], d_out=1, d_block=hp["d_block"], n_blocks=hp["n_blocks"],
                          k=hp["k"], dropout=hp["dropout"])
        fwd = lambda x: model(x_num=x).squeeze(-1)                  # (B, k)
    else:
        model = TabMLP(Xtr.shape[1], hp["hidden"], hp["dropout"])
        fwd = lambda x: model(x).unsqueeze(1)                       # (B, 1)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    g = torch.Generator().manual_seed(seed)
    best, best_state, bad, best_ep = math.inf, None, 0, 0
    for ep in range(hp["epochs"]):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(perm), hp["batch"]):
            b = perm[i:i + hp["batch"]]
            out = fwd(Xtr_t[b])
            loss = F.binary_cross_entropy_with_logits(out, ytr_t[b, None].expand_as(out), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = fwd(Xva_t)
            vl = float(F.binary_cross_entropy_with_logits(out, yva_t[:, None].expand_as(out), pos_weight=pw))
        if vl < best - 1e-5:
            best, bad, best_ep = vl, 0, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(fwd(torch.tensor(X, dtype=torch.float32))).mean(1).numpy()
    return predict, {"best_epoch": best_ep, "epochs_run": ep + 1, "best_val_loss": best,
                     "params_total": sum(p.numel() for p in model.parameters())}


def run_pheno(name, hp, pheno, features, split, cohort, seed):
    """Fit preprocessor + model di inner_train; prediksi inner_val & test."""
    lab = cohort.set_index("SUB_ID")["label"]
    X = {k: pheno.loc[v, features] for k, v in split.items()}
    y = {k: lab.loc[v].to_numpy() for k, v in split.items()}
    pre = pheno_preprocessor(features).fit(X["inner_train"])
    Z = {k: pre.transform(v) for k, v in X.items()}
    info = {"n_features_in": len(features), "n_features_out": Z["inner_train"].shape[1]}
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000).fit(Z["inner_train"], y["inner_train"])
        pred = lambda A: m.predict_proba(A)[:, 1]
        info["params_total"] = int(m.coef_.size + 1)
    elif name == "xgb":
        from xgboost import XGBClassifier
        ytr = y["inner_train"]
        m = XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
                          subsample=hp["subsample"], min_child_weight=hp["min_child_weight"],
                          scale_pos_weight=float((ytr == 0).sum() / max((ytr == 1).sum(), 1)),
                          eval_metric="logloss", early_stopping_rounds=hp["early_stopping_rounds"],
                          random_state=seed, n_jobs=4, tree_method="hist")
        m.fit(Z["inner_train"], ytr, eval_set=[(Z["inner_val"], y["inner_val"])], verbose=False)
        pred = lambda A: m.predict_proba(A)[:, 1]
        info["best_iteration"] = int(m.best_iteration)
    else:
        pred, extra = fit_torch_tabular(name if name == "tabm" else "mlp", hp, Z["inner_train"], y["inner_train"],
                                        Z["inner_val"], y["inner_val"], seed)
        info.update(extra)
    return {k: pred(Z[k]) for k in ("inner_val", "test")}, info


# ==================================================================================================
# Citra: utilitas GPU (augmentasi 3D, ekstraksi slice)
# ==================================================================================================
def augment3d(vol, mask, gen, a):
    """vol (B,X,Y,Z) float, mask bool. Rotasi kecil + translasi + intensitas; di GPU."""
    B = vol.shape[0]
    ang = (torch.rand(B, 3, generator=gen, device=vol.device) * 2 - 1) * math.radians(a["rot_deg"])
    sh = (torch.rand(B, 3, generator=gen, device=vol.device) * 2 - 1) * a["shift_vox"]
    cx, cy, cz = torch.cos(ang).unbind(1)
    sx, sy, sz = torch.sin(ang).unbind(1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    R = Rz @ Ry @ Rx
    size = torch.tensor(vol.shape[1:], device=vol.device, dtype=torch.float32).flip(0)  # (Z,Y,X) urutan grid
    t = (sh.flip(1) * 2 / size).unsqueeze(2)
    theta = torch.cat([R, t], 2)
    grid = F.affine_grid(theta, (B, 1, *vol.shape[1:]), align_corners=False)
    v = F.grid_sample(vol[:, None].float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    m = F.grid_sample(mask[:, None].float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)[:, 0] > 0.5
    sc = 1 + (torch.rand(B, 1, 1, 1, generator=gen, device=vol.device) * 2 - 1) * a["int_scale"]
    of = (torch.rand(B, 1, 1, 1, generator=gen, device=vol.device) * 2 - 1) * a["int_shift"]
    v = torch.where(m, v * sc + of, torch.zeros_like(v))
    return v, m


def make_slices(vol, mask, pos, size, jitter=0, gen=None):
    """(B,96,112,96) -> (B,S,3,size,size) dinormalisasi ImageNet. 3 slice bersebelahan = 3 kanal.
    jitter > 0 (v2, training saja): posisi tiap slice digeser acak +-jitter voxel."""
    x = torch.where(mask, ((vol.float() + 3) / 6).clamp(0, 1), torch.zeros_like(vol, dtype=torch.float32))
    out = []
    dims = {"sag": vol.shape[1], "cor": vol.shape[2], "ax": vol.shape[3]}
    for plane in ("sag", "cor", "ax"):
        for p in pos[plane]:
            if jitter:
                p = int(min(max(p + int(torch.randint(-jitter, jitter + 1, (1,), generator=gen)), 1), dims[plane] - 2))
            if plane == "sag":
                s = x[:, p - 1:p + 2]                                   # (B,3,112,96)
            elif plane == "cor":
                s = x[:, :, p - 1:p + 2].permute(0, 2, 1, 3)            # (B,3,96,96)
            else:
                s = x[:, :, :, p - 1:p + 2].permute(0, 3, 1, 2)         # (B,3,96,112)
            h, w = s.shape[-2:]
            s = F.pad(s, ((112 - w) // 2, 112 - w - (112 - w) // 2, (112 - h) // 2, 112 - h - (112 - h) // 2))
            out.append(F.interpolate(s, size=(size, size), mode="bilinear", align_corners=False))
    s = torch.stack(out, 1)
    return (s - IMNET_MEAN.to(s.device).unsqueeze(0)) / IMNET_STD.to(s.device).unsqueeze(0)


# ==================================================================================================
# Model citra
# ==================================================================================================
class Head(nn.Module):
    def __init__(self, d_in, d_emb=128, dropout=0.3, pre_dropout=0.0):
        super().__init__()
        self.emb = nn.Sequential(nn.Dropout(pre_dropout), nn.Linear(d_in, d_emb), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(d_emb, 1)


class Slice25D(nn.Module):
    """Backbone timm per slice -> rata-rata fitur antar slice -> head (embedding 128-d) -> logit subjek."""

    def __init__(self, timm_id, train_modules, freeze_bn=False):
        super().__init__()
        import timm
        self.freeze_bn = freeze_bn
        self.backbone = timm.create_model(timm_id, pretrained=True, num_classes=0)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable = []
        for name, mod in self.backbone.named_modules():
            if name in train_modules:
                for p in mod.parameters():
                    p.requires_grad = True
                self.trainable.append(name)
        assert sorted(self.trainable) == sorted(train_modules), (self.trainable, train_modules)
        self.head = Head(self.backbone.num_features)

    def train(self, mode=True):
        super().train(mode)
        if mode:                                   # bagian beku tetap eval (BN/dropout)
            self.backbone.eval()
            for name, mod in self.backbone.named_modules():
                if name in self.trainable:
                    mod.train(True)
            if self.freeze_bn:                     # v2: statistik BN pretrained dipertahankan
                for m in self.backbone.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        m.eval()
        return self

    def set_phase(self, finetune):
        """v2 LP-FT: fase linear probe (hanya head) lalu fine-tune modul terakhir."""
        for name, mod in self.backbone.named_modules():
            if name in self.trainable:
                for q in mod.parameters():
                    q.requires_grad = finetune

    def forward(self, imgs):                       # (B,S,3,H,W)
        B, S = imgs.shape[:2]
        f = self.backbone(imgs.flatten(0, 1)).view(B, S, -1).mean(1)
        e = self.head.emb(f)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        return [{"params": list(self.head.parameters()), "lr": hp["lr_head"]},
                {"params": bb, "lr": hp["lr_backbone"]}]


def sfcn_pretrained(hp):
    import importlib.util
    path = pc.PRETRAINED / "sfcn" / "sfcn.py"
    spec = importlib.util.spec_from_file_location("sfcn_upstream", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    w = pc.PRETRAINED / hp["weights"]
    assert pc.sp.sha256_file(w) == hp["weights_sha256"], "bobot SFCN tidak cocok sha256"
    net = mod.SFCN()
    sd = torch.load(w, map_location="cpu", weights_only=False)
    net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
    return net.eval()


def sfcn_input(vol, mask):
    """(B,96,112,96) 2 mm RAS -> (B,1,160,192,160) 1 mm, orientasi & crop seperti UK Biobank."""
    v = F.interpolate(vol[:, None].float(), scale_factor=2, mode="trilinear", align_corners=False)
    m = F.interpolate(mask[:, None].float(), scale_factor=2, mode="nearest") > 0.5
    # grid 1 mm: x=-96.5+j, y=-128.5+j, z=-72.5+j ; crop UKB MNI 182x218x182 [11:171,13:205,11:171]
    v, m = v[:, :, 16:176, 15:207, 11:171], m[:, :, 16:176, 15:207, 11:171]
    v, m = v.flip(2), m.flip(2)                    # RAS -> LAS (sumbu x seperti array FSL/UKB)
    vmin = torch.where(m, v, torch.full_like(v, 1e9)).amin(dim=(2, 3, 4), keepdim=True)
    v = torch.where(m, v - vmin + 1e-3, torch.zeros_like(v))
    return v / v.mean(dim=(2, 3, 4), keepdim=True)


class SFCNTail(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.conv_4 = net.feature_extractor.conv_4
        self.conv_5 = net.feature_extractor.conv_5
        self.head = Head(64, pre_dropout=0.5)

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()                           # statistik BN UKB dipertahankan (batch kecil)
        return self

    def forward(self, f):                          # (B,256,10,12,10)
        x = self.conv_5(self.conv_4(f)).mean(dim=(2, 3, 4))
        e = self.head.emb(x)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.head.parameters()), "lr": hp["lr_head"]},
                {"params": list(self.conv_4.parameters()) + list(self.conv_5.parameters()), "lr": hp["lr_backbone"]}]


class SliceFeatHead(nn.Module):
    """Head untuk fitur DINOv2 per slice (S,768): MLP per slice -> rata-rata -> logit."""

    def __init__(self, d_in):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.head = Head(d_in)

    def forward(self, f):                          # (B,S,768)
        e = self.head.emb(self.norm(f.float())).mean(1)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.parameters()), "lr": hp["lr"]}]


# ==================================================================================================
# Loop training generik dengan checkpoint per epoch
# ==================================================================================================
def warmup_cosine(hp, group):
    """Faktor LR per epoch. group 0 = head; group 1 = backbone (0 selama lp_epochs, lalu warmup + cosine)."""
    w, E = hp["warmup"], hp["epochs"]
    start = hp.get("lp_epochs", 0) if group == 1 else 0

    def f(e):
        if e < start:
            return 0.0
        e2, n = e - start, max(E - start, 1)
        if e2 < w:
            return (e2 + 1) / w
        return 0.5 * (1 + math.cos(math.pi * min((e2 - w) / max(n - w, 1), 1.0)))
    return f


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["py"]); np.random.set_state(s["np"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def recalibrate_bn(model, batches):
    """Bugfix BN: reset running stats lalu hitung ulang sebagai rerata kumulatif atas batches() (inner_train tanpa
    augmentasi). Hanya BN yang diubah; dropout dan bobot tidak tersentuh. Model dikembalikan dalam mode eval."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    mom = [m.momentum for m in bns]
    model.eval()
    for m in bns:
        m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
        for x in batches():
            model(x)
    for m, mo in zip(bns, mom):
        m.momentum = mo
    model.eval()


def train_loop(model, hp, batches_fn, eval_fn, ckpt_dir, seed, stop, budget_s, on_epoch, ytr, bn_recal_fn=None):
    """batches_fn(epoch) -> iterator (x, y) train; eval_fn(model) -> (val_loss, ...). Checkpoint tiap epoch:
    last.pt (model, optimizer, scheduler, scaler, epoch, best, rng, waktu) + best.pt (bobot terbaik).
    bn_recal_fn(model): bila diberikan, dipanggil sebelum tiap evaluasi (bugfix BN; best.pt menyimpan statistik ini)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.param_groups(hp), weight_decay=hp["wd"])
    if "warmup" in hp:                             # v2: warmup linear lalu cosine; backbone 0 selama fase LP
        sched = torch.optim.lr_scheduler.LambdaLR(opt, [warmup_cosine(hp, i) for i in range(len(opt.param_groups))])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=DEV.type == "cuda")
    pw = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)), device=DEV)
    state = {"epoch": -1, "best": math.inf, "best_epoch": -1, "bad": 0, "elapsed": 0.0, "history": []}
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); set_rng_state(ck["rng"]); state = ck["state"]
        LOG.info("Resume dari checkpoint epoch %d (best epoch %d, val loss %.4f)", state["epoch"] + 1,
                 state["best_epoch"] + 1, state["best"])
    else:
        seed_all(seed)
    start_ep = state["epoch"] + 1
    for ep in range(start_ep, hp["epochs"]):
        if state["bad"] >= hp["patience"] and ep >= hp.get("min_epochs", 0):
            break
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats() if DEV.type == "cuda" else None
        if "lp_epochs" in hp and hasattr(model, "set_phase"):
            model.set_phase(finetune=ep >= hp["lp_epochs"])
        model.train()
        tl, nb = 0.0, 0
        for x, y in batches_fn(ep):
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                logit, _ = model(x)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        sched.step()
        if bn_recal_fn is not None:
            bn_recal_fn(model)
        res = eval_fn(model)
        vl, vauc = res if isinstance(res, tuple) else (res, float("nan"))
        dt = time.time() - t0
        state["elapsed"] += dt
        vram = torch.cuda.max_memory_allocated() / 2**20 if DEV.type == "cuda" else 0
        k = hp.get("smooth", 1)                    # v2: kriteria = rerata k val loss terakhir
        crit = float(np.mean([h["val_loss"] for h in state["history"][-(k - 1):]] + [vl])) if k > 1 and state["history"] else vl
        state["history"].append({"epoch": ep + 1, "train_loss": tl / max(nb, 1), "val_loss": vl, "val_auc": vauc,
                                 "val_crit": crit, "seconds": dt, "vram_mib": vram,
                                 "lr": [g["lr"] for g in opt.param_groups]})
        vl = crit
        if vl < state["best"] - 1e-5:
            state.update(best=vl, best_epoch=ep, bad=0)
            save_torch_atomic(model.state_dict(), ckpt_dir / "best.pt")
        else:
            state["bad"] += 1
        state["epoch"] = ep
        save_torch_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                           "scaler": scaler.state_dict(), "rng": rng_state(), "state": state}, last)
        on_epoch(ep, dt, vl, vram, state)
        if stop():
            raise StopRequested()
        if state["elapsed"] > budget_s:
            raise BudgetExceeded(f"budget {budget_s / 60:.0f} menit terlampaui pada epoch {ep + 1}")
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False))
    return state


# ==================================================================================================
# v2: CNN dari nol, head fitur beku, probe linear, model tabel dengan inner CV
# ==================================================================================================
class Scratch2D(nn.Module):
    """CNN 2D kecil dari nol: blok (conv-BN-ReLU x2, maxpool) -> GAP per slice -> rata-rata slice -> head."""

    def __init__(self, channels, dropout):
        super().__init__()
        layers, cin = [], 3
        for c in channels:
            layers += [nn.Conv2d(cin, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2), nn.Dropout2d(0.1)]
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = Head(cin, dropout=dropout)

    def forward(self, imgs):
        B, S = imgs.shape[:2]
        f = self.features(imgs.flatten(0, 1)).mean(dim=(2, 3)).view(B, S, -1).mean(1)
        e = self.head.emb(f)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.parameters()), "lr": hp["lr"]}]


class SE3d(nn.Module):
    def __init__(self, c, r):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=(2, 3, 4)))
        return x * w[:, :, None, None, None]


class Scratch3DSE(nn.Module):
    """CNN 3D kecil + Squeeze-Excitation dari nol: input (B,1,96,112,96)."""

    def __init__(self, channels, se_ratio, dropout):
        super().__init__()
        layers, cin = [], 1
        for i, c in enumerate(channels):
            layers += [nn.Conv3d(cin, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(inplace=True),
                       nn.Conv3d(c, c, 3, padding=1, bias=False), nn.BatchNorm3d(c), nn.ReLU(inplace=True),
                       SE3d(c, se_ratio)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool3d(2))
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = Head(cin, dropout=dropout)

    def forward(self, v):
        f = self.features(v).mean(dim=(2, 3, 4))
        e = self.head.emb(f)
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.parameters()), "lr": hp["lr"]}]


class FeatHead(nn.Module):
    """Head untuk vektor fitur beku yang sudah distandarkan (SFCN 64-d / DINOv2 768-d)."""

    def __init__(self, d_in, dropout):
        super().__init__()
        self.head = Head(d_in, dropout=dropout, pre_dropout=0.2)

    def forward(self, x):
        e = self.head.emb(x.float())
        return self.head.out(e).squeeze(1), e

    def param_groups(self, hp):
        return [{"params": list(self.parameters()), "lr": hp["lr"]}]


def fit_probe(Xtr, ytr, Xva, yva, C_grid, seed):
    """Linear probe: StandardScaler (fit train) + LogReg; C dipilih dari AUC inner val (bukan test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    best = None
    for C in C_grid:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed).fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yva, m.predict_proba(sc.transform(Xva))[:, 1])
        if best is None or auc > best[0] + 1e-9:
            best = (auc, C, m)
    _, C, m = best
    return (lambda X: m.predict_proba(sc.transform(X))[:, 1]), {"C": C, "auc_inner_val": best[0]}, sc


class TabPre:
    """Preprocessing tabel di-fit pada train: fenotipik (ColumnTransformer) atau ROI (median -> [ComBat] -> scaler)."""

    def __init__(self, kind, features, combat=False):
        self.kind, self.features, self.combat = kind, features, combat

    def fit(self, X, site=None, cov=None):
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        if self.kind == "pheno":
            self.ct = pheno_preprocessor(self.features).fit(X[self.features])
            return self
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
        Z = self.imp.transform(X)
        if self.combat:
            self.cm = pc.combat_fit(Z, site, cov)
            Z, _ = pc.combat_apply(self.cm, Z, site, cov)
        self.sc = StandardScaler().fit(Z)
        return self

    def transform(self, X, site=None, cov=None):
        if self.kind == "pheno":
            return self.ct.transform(X[self.features])
        Z = self.imp.transform(X)
        self.unseen = 0
        if self.combat:
            Z, un = pc.combat_apply(self.cm, Z, site, cov)
            self.unseen = int(un.sum())
        return self.sc.transform(Z)


def fit_tab_model(name, hp, Z, y, seed):
    """Fit model tabel pada Z (sudah diproses). Model iteratif memakai 20% Z (stratified) untuk early stopping."""
    from sklearn.model_selection import train_test_split
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=5000).fit(Z, y)
        return lambda A: m.predict_proba(A)[:, 1]
    tr, es = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        m = XGBClassifier(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"], learning_rate=hp["learning_rate"],
                          subsample=hp["subsample"], min_child_weight=hp["min_child_weight"],
                          scale_pos_weight=float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)),
                          eval_metric="logloss", early_stopping_rounds=hp["early_stopping_rounds"],
                          random_state=seed, n_jobs=4, tree_method="hist")
        m.fit(Z[tr], y[tr], eval_set=[(Z[es], y[es])], verbose=False)
        return lambda A: m.predict_proba(A)[:, 1]
    pred, _ = fit_torch_tabular("tabm" if name == "tabm" else "mlp", hp, Z[tr], y[tr], Z[es], y[es], seed)
    return pred


def run_tabular_v2(base, hp, X, cohort, split, seed, kind, features, combat, cov, n_inner):
    """Train luar T = inner_train + inner_val. OOF inner CV (k-fold) pada T -> seleksi & stacking;
    model akhir di-fit ulang pada T -> test. Semua preprocessing (impute, ComBat, scaler) di-fit di data latih."""
    from sklearn.model_selection import StratifiedKFold
    lab = cohort.set_index("SUB_ID")["label"]
    site_of = cohort.set_index("SUB_ID")["SITE_ID"]
    T = np.concatenate([split["inner_train"], split["inner_val"]])
    yT = lab.loc[T].to_numpy()
    oof = np.full(len(T), np.nan)
    unseen = 0
    for j, (a, b) in enumerate(StratifiedKFold(n_inner, shuffle=True, random_state=seed).split(T, yT)):
        ta, tb = T[a], T[b]
        pre = TabPre(kind, features, combat).fit(X.loc[ta], site_of.loc[ta].to_numpy(), cov.loc[ta].reset_index(drop=True))
        Za = pre.transform(X.loc[ta], site_of.loc[ta].to_numpy(), cov.loc[ta].reset_index(drop=True))
        Zb = pre.transform(X.loc[tb], site_of.loc[tb].to_numpy(), cov.loc[tb].reset_index(drop=True))
        unseen += getattr(pre, "unseen", 0)
        oof[b] = fit_tab_model(base, hp, Za, yT[a], seed + j)(Zb)
    pre = TabPre(kind, features, combat).fit(X.loc[T], site_of.loc[T].to_numpy(), cov.loc[T].reset_index(drop=True))
    ZT = pre.transform(X.loc[T], site_of.loc[T].to_numpy(), cov.loc[T].reset_index(drop=True))
    te = split["test"]
    Zt = pre.transform(X.loc[te], site_of.loc[te].to_numpy(), cov.loc[te].reset_index(drop=True))
    unseen_test = getattr(pre, "unseen", 0)
    pt = fit_tab_model(base, hp, ZT, yT, seed)(Zt)
    oof_s = pd.Series(oof, index=T)
    probs = {"inner_oof": oof, "inner_val": oof_s.loc[split["inner_val"]].to_numpy(), "test": pt}
    ids = {"inner_oof": T, "inner_val": split["inner_val"], "test": te}
    return probs, ids, {"n_features_out": int(ZT.shape[1]), "combat_unseen_inner": unseen, "combat_unseen_test": unseen_test}
