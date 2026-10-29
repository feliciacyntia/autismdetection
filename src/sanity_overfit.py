"""Sanity check sebelum data penuh: overfit 16 subjek train (8 ASD + 8 TD, fold 0 inner_train).
Tanpa regularisasi (dropout 0, wd 0, tanpa augmentasi/jitter), tanpa early stopping, LR konstan, N epoch.
Arsitektur, modul yang dilatih, BN beku, batch, dan AMP sama dengan pipeline v2 (config terkunci tidak diubah).
Lulus: loss eval-mode pada 16 subjek < 0,05 dan akurasi 16/16."""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pilot_common as pc
import pilot_models as pm

IMG = ["effnet_b0", "resnet18", "scratch2d", "convnextv2_t", "scratch3d", "sfcn_head", "dinov2_head"]


def pick16(rdir, cohort, seed=42):
    sp = json.loads((rdir / "splits.json").read_text(encoding="utf-8"))
    ids = np.array(sp["0"]["inner_train"] if "0" in sp else sp[0]["inner_train"])
    lab = cohort.set_index("SUB_ID")["label"].loc[ids].to_numpy()
    rng = np.random.default_rng(seed)
    out = np.concatenate([rng.choice(ids[lab == c], 8, replace=False) for c in (1, 0)])
    return out, cohort.set_index("SUB_ID")["label"].loc[out].to_numpy()


def build(name, hp, feats=None):
    k = hp["kind"]
    if k == "2d":
        m = pm.Slice25D(hp["timm"], hp["train_modules"], freeze_bn=hp.get("freeze_bn", False))
        m.set_phase(finetune=True)                         # tanpa fase LP: langsung semua modul yang dilatih
    elif k == "scratch2d":
        m = pm.Scratch2D(hp["channels"], 0.0)
    elif k == "scratch3d":
        m = pm.Scratch3DSE(hp["channels"], hp["se_ratio"], 0.0)
    else:
        m = pm.FeatHead(feats.shape[1], 0.0)
    for mod in m.modules():                                # tanpa regularisasi
        if isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            mod.p = 0.0
    return m.to(pm.DEV)


def run(name, hp, x_fn, y, epochs, amp, log):
    pm.seed_all(42)
    feats = x_fn(None) if hp["kind"] == "fhead" else None
    model = build(name, hp, feats)
    groups = model.param_groups(hp)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and pm.DEV.type == "cuda")
    yt = torch.tensor(y, dtype=torch.float32, device=pm.DEV)
    B, n = hp["batch"], len(y)
    hist = []
    t0 = time.time()

    def eval_loss():
        if hp.get("bn_recal"):                         # sama dengan pipeline: statistik BN dari data latih tanpa augmentasi
            pm.recalibrate_bn(model, lambda: (x_fn(torch.arange(i, min(i + B, n))) for i in range(0, n, B)))
        model.eval()
        with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16, enabled=amp):
            lg = torch.cat([model(x_fn(torch.arange(i, min(i + B, n))))[0].float() for i in range(0, n, B)])
        return float(F.binary_cross_entropy_with_logits(lg, yt)), float(((lg > 0).float() == yt).float().mean())

    for ep in range(epochs):
        model.train()
        g = torch.Generator().manual_seed(ep)
        perm = torch.randperm(n, generator=g)
        tl, nb = 0.0, 0
        for i in range(0, n, B):
            b = perm[i:i + B]
            with torch.autocast(pm.DEV.type, dtype=torch.float16, enabled=amp):
                lg, _ = model(x_fn(b))
                loss = F.binary_cross_entropy_with_logits(lg.float(), yt[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        rec = {"epoch": ep + 1, "train_loss": tl / nb}
        if (ep + 1) % 10 == 0 or ep == 0:
            rec["eval_loss"], rec["eval_acc"] = eval_loss()
            log(f"{name:13s} ep {ep + 1:3d} | train {rec['train_loss']:.4f} | eval {rec['eval_loss']:.4f} acc {rec['eval_acc']:.2f}")
        hist.append(rec)
    el, ea = eval_loss()
    diag = {}
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training is False]
    trained_bn = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.weight is not None
                  and m.weight.requires_grad]
    if trained_bn and not getattr(model, "freeze_bn", False):
        # diagnostik BN: (a) eval dengan statistik batch (ukuran batch sama), (b) kalibrasi ulang running stats
        def bn_mode(train):
            for m in trained_bn:
                m.train(train)
        model.eval(); bn_mode(True)
        saved = {id(m): (m.running_mean.clone(), m.running_var.clone(), m.momentum) for m in trained_bn}
        with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16, enabled=amp):
            lg = torch.cat([model(x_fn(torch.arange(i, min(i + B, n))))[0].float() for i in range(0, n, B)])
        diag["eval_batchstats"] = (float(F.binary_cross_entropy_with_logits(lg, yt)), float(((lg > 0).float() == yt).float().mean()))
        for m in trained_bn:
            m.reset_running_stats(); m.momentum = None        # rata-rata kumulatif atas seluruh data train
        with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16, enabled=amp):
            for i in range(0, n, B):
                model(x_fn(torch.arange(i, min(i + B, n))))
        diag["eval_recal"] = eval_loss()
        for m in trained_bn:
            m.running_mean.copy_(saved[id(m)][0]); m.running_var.copy_(saved[id(m)][1]); m.momentum = saved[id(m)][2]
        log(f"    BN diag {name}: eval statistik batch loss {diag['eval_batchstats'][0]:.4f} acc {diag['eval_batchstats'][1]:.2f} | "
            f"eval setelah kalibrasi ulang running stats loss {diag['eval_recal'][0]:.4f} acc {diag['eval_recal'][1]:.2f}")
    del bns
    return {"model": name, "amp": amp, "epochs": epochs, "train_loss_ep1": hist[0]["train_loss"], "bn_diag": diag,
            "train_loss_last": hist[-1]["train_loss"], "eval_loss": el, "eval_acc": ea,
            "lulus": bool(el < 0.05 and ea == 1.0), "menit": (time.time() - t0) / 60,
            "lr": [gr["lr"] for gr in groups], "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(IMG))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = pc.CONFIG_V2
    rdir = pc.results_dir("pilot200", "v2")
    out = rdir / "sanity"; out.mkdir(exist_ok=True)
    cohort, _ = pc.load_cohort("pilot200")
    ids, y = pick16(rdir, cohort)
    V, M = pc.load_volumes(ids)
    V, M = torch.from_numpy(V), torch.from_numpy(M)
    pos = pc.slice_positions()
    logf = open(out / f"overfit{a.tag}.log", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    log(f"=== overfit 16 subjek | {time.strftime('%Y-%m-%d %H:%M')} | label ASD={int(y.sum())} TD={int((1 - y).sum())} | "
        f"AMP={not a.no_amp} | epoch {a.epochs}")
    res = []
    for name in a.models.split(","):
        hp = cfg["models"][name]
        k = hp["kind"]
        if k == "fhead":
            d = rdir / "cache" / hp["source"]
            raw = np.stack([np.load(d / f"{int(s)}.npy").astype(np.float32) for s in ids])
            raw = raw.mean(1) if raw.ndim == 3 else raw
            Z = torch.tensor((raw - raw.mean(0)) / (raw.std(0) + 1e-6), device=pm.DEV)
            x_fn = (lambda Z: lambda b: Z if b is None else Z[b])(Z)
        else:
            def x_fn(b, k=k, hp=hp):
                v, m = V[b].to(pm.DEV).float(), M[b].to(pm.DEV)
                if k == "scratch3d":
                    return torch.where(m, v, torch.zeros_like(v))[:, None]
                return pm.make_slices(v, m, pos, hp.get("img_size", cfg["slices"]["size"]))
        r = run(name, hp, x_fn, y, a.epochs, not a.no_amp, log)
        log(f"--> {name}: train ep1 {r['train_loss_ep1']:.4f} -> akhir {r['train_loss_last']:.4f} | eval loss "
            f"{r['eval_loss']:.4f} acc {r['eval_acc']:.2f} | {'LULUS' if r['lulus'] else 'GAGAL'} | {r['menit']:.1f} menit")
        res.append(r)
        torch.cuda.empty_cache()
    p = out / f"overfit{a.tag}.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(old + res, indent=1), encoding="utf-8"); tmp.replace(p)


if __name__ == "__main__":
    main()
