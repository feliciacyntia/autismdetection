"""06 — Runner PILOT v2 (perbaikan training citra, fitur ROI + ComBat, CNN dari nol, fusion & ensemble).

    .venv-model\\Scripts\\python.exe src\\run_pilot_v2.py --subset pilot200 --resume [--stop-at 07:00]

Infrastruktur sama dengan run_pilot.py (status per unit, checkpoint per epoch, resume, STOP, --stop-at, lock basi
otomatis, cegah sleep). Hasil: <results>/<subset>_v2/. Kode keluar sama dengan run_pilot.py.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402
import run_pilot as rp  # noqa: E402

LOG = logging.getLogger("pilot")


def lg(a):
    a = np.clip(np.asarray(a, float), 1e-6, 1 - 1e-6)
    return np.log(a / (1 - a))


class RunnerV2(rp.Runner):
    VERSION = "v2"

    # ---------------- daftar unit ----------------
    def units(self):
        out, folds = [], sorted(self.splits)
        M = self.cfg["models"]
        for name in self.cfg["order"]:
            if name.startswith("cache_"):
                out.append(name)
            elif name == "EMBCOMBAT":
                out += [f"embcb_{m}:{k}" for m in self.cfg["modality"]["img25d"] + self.cfg["modality"]["img3d"] for k in folds]
            elif name == "FUSION":
                out += [f"fusion_{s}:{k}" for s in self.cfg["fusion"]["scenarios"] for k in folds]
            elif name == "ENSEMBLE":
                out += [f"fusion_ENS:{k}" for k in folds]
            else:
                assert name in M, name
                out += [f"{name}:{k}" for k in folds]
        if self.args.only:
            keep = self.args.only.split(",")
            out = [u for u in out if any(u == k or u.startswith(k.rstrip("*")) for k in keep)]
        return out

    def dispatch(self, u, model, fold, is_fusion):
        if u.startswith("cache_"):
            return self.run_cache_v2(u, u[len("cache_"):])
        if is_fusion:
            return self.run_fusion_v2(u, model.split("_", 1)[1], fold)
        if model.startswith("embcb_"):
            return self.run_embcombat(u, model[len("embcb_"):], fold)
        kind = self.cfg["models"][model]["kind"]
        if kind in ("pheno", "roi"):
            return self.run_tab_v2(u, model, fold)
        if kind == "probe":
            return self.run_probe(u, model, fold)
        return self.run_image_v2(u, model, fold)

    # ---------------- data bantu ----------------
    def cov(self, ids):
        return self.pheno.loc[ids, ["AGE_AT_SCAN", "SEX"]].reset_index(drop=True)

    def roi_table(self):
        return pd.read_csv(self.cache_dir("roi") / "roi_features.csv").set_index("SUB_ID")

    def feat(self, source, ids):
        d = self.cache_dir(source)
        if source == "dino768":
            return np.stack([np.load(d / f"{int(s)}.npy").astype(np.float32).mean(0) for s in ids])
        return np.stack([np.load(d / f"{int(s)}.npy").astype(np.float32) for s in ids])

    # ---------------- cache ----------------
    def run_cache_v2(self, unit, name):
        import torch
        import pilot_models as pm
        d = self.cache_dir(name.replace("dino768", "dino768")); d.mkdir(parents=True, exist_ok=True)
        ids = self.cohort["SUB_ID"].to_numpy()
        t = time.time()
        info = {}
        if name == "roi":
            mc = self.cfg["roi"]["min_voxels"]
            df = pc.roi_features(ids, mc)
            pc.sp.write_csv_atomic(d / "roi_features.csv", df.reset_index())
            info = {"n_features": int(df.shape[1]), "nan_frac": float(df.isna().mean().mean())}
            LOG.info("Fitur ROI: %d subjek x %d fitur | NaN %.4f", len(df), df.shape[1], info["nan_frac"])
        else:
            V = self.volumes()
            torch.cuda.reset_peak_memory_stats()
            if name == "sfcn64":
                net = pm.sfcn_pretrained(self.cfg["sfcn"]).to(pm.DEV).eval()
                ages = []
            else:
                import timm
                net = timm.create_model(self.cfg["dinov2"]["timm"], pretrained=True, num_classes=0,
                                        img_size=self.cfg["dinov2"]["img_size"]).to(pm.DEV).eval()
            for i, s in enumerate(ids):
                v, m = self._tensors([s], "vol")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    v, m = v.to(pm.DEV), m.to(pm.DEV)
                    if name == "sfcn64":
                        f = net.feature_extractor(pm.sfcn_input(v, m))                 # (1,64,5,6,5)
                        logp = net.classifier(f).float().view(1, -1)                   # 40 bin usia (42..82)
                        age = float((torch.softmax(logp, 1) * (torch.arange(40, device=logp.device) + 42.5)).sum())
                        ages.append({"SUB_ID": int(s), "brain_age_sfcn": age})
                        out = f.float().mean(dim=(2, 3, 4))[0].cpu().numpy()
                    else:
                        imgs = pm.make_slices(v, m, self.pos, self.cfg["slices"]["size"])[0]
                        tok = net.forward_features(imgs)
                        out = torch.cat([tok[:, 0], tok[:, 1:].mean(1)], 1).float().cpu().numpy().astype(np.float16)
                if not (d / f"{int(s)}.npy").exists():
                    pc.sp.save_npy_atomic(d / f"{int(s)}.npy", out)
                if (i + 1) % 50 == 0:
                    LOG.info("Cache %s: %d/%d", name, i + 1, len(ids))
                if self.stop():
                    raise pm.StopRequested()
            info["vram_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            if name == "sfcn64":
                from scipy.stats import pearsonr, spearmanr
                a = pd.DataFrame(ages).set_index("SUB_ID").join(self.pheno["AGE_AT_SCAN"])
                pc.sp.write_csv_atomic(d / "brain_age_sfcn.csv", a.reset_index())
                san = {}
                for lbl, q in (("semua", a), ("usia>=18", a[a.AGE_AT_SCAN >= 18]), ("usia>=40", a[a.AGE_AT_SCAN >= 40])):
                    if len(q) >= 5:
                        san[lbl] = {"n": len(q), "pearson_r": float(pearsonr(q.brain_age_sfcn, q.AGE_AT_SCAN)[0]),
                                    "spearman_r": float(spearmanr(q.brain_age_sfcn, q.AGE_AT_SCAN)[0])}
                info["sanity_brain_age"] = san
                LOG.info("Cek kewajaran SFCN (brain age vs AGE_AT_SCAN, tanpa label diagnosis): %s", san)
            del net
            torch.cuda.empty_cache()
        info["seconds"] = time.time() - t
        pc.sp.write_json_atomic(pc.unit_dir(self.rdir, unit) / "cache_done.json", info)
        return {"seconds": info["seconds"], "vram_peak_mib": info.get("vram_peak_mib")}

    # ---------------- tabel: fenotipik & ROI ----------------
    def run_tab_v2(self, unit, name, fold):
        import pilot_models as pm
        hp = self.cfg["models"][name]
        t = time.time()
        if hp["kind"] == "pheno":
            X, base, kind, feats, combat = self.pheno, name, "pheno", self.features, False
        else:
            X = self.roi_table()
            base, kind, feats, combat = hp["base"], "roi", list(X.columns), hp["combat"]
        bhp = self.cfg["models"][base]
        cov = self.pheno[["AGE_AT_SCAN", "SEX"]]
        probs, ids, extra = pm.run_tabular_v2(base, bhp, X, self.cohort, self.splits[fold], self.cfg["seed"] + fold,
                                              kind, feats, combat, cov, self.cfg["tabular"]["inner_cv_folds"])
        info = {"seconds": time.time() - t, "base": base, "combat": combat, **extra}
        m = rp.save_unit(self.rdir, unit, self.cohort, fold, probs, ids, {"info": info})
        return {"seconds": info["seconds"]}

    # ---------------- probe linear pada fitur beku ----------------
    def run_probe(self, unit, name, fold):
        import pilot_models as pm
        hp = self.cfg["models"][name]
        sp_ = self.splits[fold]
        lab = self.cohort.set_index("SUB_ID")["label"]
        X = {k: self.feat(hp["source"], v) for k, v in sp_.items()}
        y = {k: lab.loc[v].to_numpy() for k, v in sp_.items()}
        t = time.time()
        pred, extra, sc = pm.fit_probe(X["inner_train"], y["inner_train"], X["inner_val"], y["inner_val"],
                                       hp["C_grid"], self.cfg["seed"] + fold)
        probs = {k: pred(X[k]) for k in ("inner_val", "test")}
        emb = {k: sc.transform(X[k]).astype(np.float32) for k in X}
        info = {"seconds": time.time() - t, "n_features": int(X["inner_train"].shape[1]), **extra}
        rp.save_unit(self.rdir, unit, self.cohort, fold, probs, sp_, {"info": info}, emb=emb)
        return {"seconds": info["seconds"]}

    # ---------------- model citra (pretrained LP-FT, dari nol, head fitur beku) ----------------
    def run_image_v2(self, unit, name, fold):
        import torch
        import pilot_models as pm
        hp = self.cfg["models"][name]
        kind = hp["kind"]
        sp_ = self.splits[fold]
        lab = self.cohort.set_index("SUB_ID")["label"]
        y = {k: torch.tensor(lab.loc[v].to_numpy(), dtype=torch.float32) for k, v in sp_.items()}
        seed = self.cfg["seed"] + fold
        pm.seed_all(seed)
        if kind == "2d":
            model = pm.Slice25D(hp["timm"], hp["train_modules"], freeze_bn=hp.get("freeze_bn", False))
        elif kind == "scratch2d":
            model = pm.Scratch2D(hp["channels"], hp["dropout"])
        elif kind == "scratch3d":
            model = pm.Scratch3DSE(hp["channels"], hp["se_ratio"], hp["dropout"])
        else:
            from sklearn.preprocessing import StandardScaler
            F_ = {k: self.feat(hp["source"], v) for k, v in sp_.items()}
            sc = StandardScaler().fit(F_["inner_train"])
            feats = {k: torch.tensor(sc.transform(v), dtype=torch.float32) for k, v in F_.items()}
            model = pm.FeatHead(feats["inner_train"].shape[1], hp["dropout"])
        if kind in ("2d", "scratch2d", "scratch3d"):
            # indeks baris ke array volume bersama (tanpa salinan per partisi; hemat RAM di data penuh, nilai identik)
            Vall = self.volumes()
            data = {k: np.asarray([Vall["idx"][int(s)] for s in v]) for k, v in sp_.items()}
        model = model.to(pm.DEV)
        ptot, ptr = pm.count_params(model)
        if kind == "2d":                            # parameter yang dilatih setelah fase LP
            ptr = sum(q.numel() for q in model.head.parameters()) + sum(
                q.numel() for n_, mod in model.backbone.named_modules() if n_ in model.trainable for q in mod.parameters())
        B = hp["batch"]
        gens = {}

        def prep(part, idx, train):
            if kind == "fhead":
                return feats[part][idx].to(pm.DEV)
            rows = data[part][np.asarray(idx)]
            v, m = torch.from_numpy(Vall["v"][rows]).to(pm.DEV).float(), torch.from_numpy(Vall["m"][rows]).to(pm.DEV)
            if train:
                v, m = pm.augment3d(v, m, gens["aug"], self.cfg["augment"])
            if kind == "scratch3d":
                return torch.where(m, v, torch.zeros_like(v))[:, None]
            size = hp.get("img_size", self.cfg["slices"]["size"])
            return pm.make_slices(v, m, self.pos, size, jitter=self.cfg["slice_jitter_vox"] if train else 0,
                                  gen=gens.get("jit"))

        def batches(ep):
            g = torch.Generator().manual_seed(seed * 1000 + ep)
            gens["aug"] = torch.Generator(device=pm.DEV).manual_seed(seed * 1000 + ep)
            gens["jit"] = torch.Generator().manual_seed(seed * 7919 + ep)
            perm = torch.randperm(len(sp_["inner_train"]), generator=g)
            for i in range(0, len(perm), B):
                b = perm[i:i + B]
                yield prep("inner_train", b, True), y["inner_train"][b].to(pm.DEV)

        def predict(part):
            model.eval()
            P, E = [], []
            with torch.no_grad(), torch.autocast(pm.DEV.type, dtype=torch.float16):
                for i in range(0, len(sp_[part]), B):
                    b = torch.arange(i, min(i + B, len(sp_[part])))
                    lg_, e = model(prep(part, b, False))
                    P.append(lg_.float().cpu()); E.append(e.float().cpu())
            return torch.cat(P), torch.cat(E)

        ytr = y["inner_train"].numpy()
        pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

        def evaluate(_m):
            from sklearn.metrics import roc_auc_score
            lg_, _ = predict("inner_val")
            loss = float(torch.nn.functional.binary_cross_entropy_with_logits(lg_, y["inner_val"], pos_weight=torch.tensor(pw)))
            return loss, float(roc_auc_score(y["inner_val"].numpy(), lg_.numpy()))

        first = {}

        def on_epoch(ep, dt, vl, vram, st):
            h = st["history"][-1]
            LOG.info("%s | epoch %d/%d | %.1f s | train %.4f | val %.4f (halus %.4f) | AUC val %.3f | best %d | VRAM %.0f MiB",
                     unit, ep + 1, hp["epochs"], dt, h["train_loss"], h["val_loss"], h["val_crit"], h["val_auc"],
                     st["best_epoch"] + 1, vram)
            if not first and fold == min(self.splits):
                first["s"] = dt
                LOG.info("ESTIMASI %s: ±%.1f s/epoch -> maks ±%.0f menit untuk %d fold", name, dt,
                         dt * hp["epochs"] * len(self.splits) / 60, len(self.splits))
            pc.update_status(self.rdir, unit, last_epoch=ep + 1, max_epochs=hp["epochs"], best_epoch=st["best_epoch"] + 1,
                             best_val_loss=st["best"], seconds=st["elapsed"], vram_peak_mib=vram,
                             params_total=ptot, params_trainable=ptr)
            rp.write_progress(self.rdir, unit=unit, epoch=ep + 1, max_epochs=hp["epochs"], epoch_s=round(dt, 1),
                              eta_unit_min=round(dt * (hp["epochs"] - ep - 1) / 60, 1))

        ckpt = pc.unit_dir(self.rdir, unit) / "ckpt"
        budget = hp.get("budget_min", 60) * 60 * self.cfg.get("budget_scale", 1.0) / len(self.splits)
        recal = None
        if hp.get("bn_recal"):                      # bugfix BN: statistik dari inner_train tanpa augmentasi
            def recal(m):
                n_tr = len(sp_["inner_train"])
                pm.recalibrate_bn(m, lambda: (prep("inner_train", torch.arange(i, min(i + B, n_tr)), False)
                                              for i in range(0, n_tr, B)))
        st = pm.train_loop(model, hp, batches, evaluate, ckpt, seed, self.stop, budget, on_epoch, ytr, bn_recal_fn=recal)
        if recal is not None:                       # juga sebelum prediksi akhir (bobot best.pt)
            recal(model)
        probs, embs = {}, {}
        for part in ("inner_train", "inner_val", "test"):
            lg_, e = predict(part)
            probs[part] = torch.sigmoid(lg_).numpy(); embs[part] = e.numpy().astype(np.float32)
        # kriteria "sudah belajar" (train & inner val saja)
        h = st["history"]
        p0 = float(ytr.mean())
        yv = y["inner_val"].numpy()
        const_loss = float(np.mean(pw * yv * -np.log(p0) + (1 - yv) * -np.log(1 - p0)))
        auc_best = h[st["best_epoch"]]["val_auc"]
        learned = bool(h[-1]["train_loss"] < 0.98 * h[0]["train_loss"] and (st["best"] < const_loss or auc_best >= 0.55))
        vram = max(x["vram_mib"] for x in h)
        info = {"best_epoch": st["best_epoch"] + 1, "epochs_run": st["epoch"] + 1, "best_val_loss": st["best"],
                "const_val_loss": const_loss, "val_auc_best_epoch": auc_best, "learned": learned,
                "seconds": st["elapsed"], "vram_peak_mib": vram, "params_total": ptot, "params_trainable": ptr,
                "s_per_epoch": float(np.mean([x["seconds"] for x in h])), "history": h,
                "trainable_modules": hp.get("train_modules", "semua (dari nol / head)"), "kind": kind}
        rp.save_unit(self.rdir, unit, self.cohort, fold, probs, sp_, {"info": info}, emb=embs)
        LOG.info("%s: sudah belajar = %s (train %.3f->%.3f | val halus terbaik %.4f vs konstan %.4f | AUC val %.3f)",
                 unit, learned, h[0]["train_loss"], h[-1]["train_loss"], st["best"], const_loss, auc_best)
        del model
        torch.cuda.empty_cache()
        return {"seconds": st["elapsed"], "vram_peak_mib": vram, "params_total": ptot, "params_trainable": ptr,
                "best_epoch": st["best_epoch"] + 1, "best_val_loss": st["best"]}

    # ---------------- ComBat pada embedding (ablasi) + site learning ----------------
    def run_embcombat(self, unit, img, fold):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.preprocessing import StandardScaler
        import pilot_models as pm
        src = f"{img}:{fold}"
        if not pc.unit_done_valid(self.rdir, src):
            raise RuntimeError(f"SKIP: {src} belum selesai")
        e = np.load(pc.unit_dir(self.rdir, src) / "emb.npz")
        sp_ = self.splits[fold]
        c = self.cohort.set_index("SUB_ID")
        E = {k: e[f"emb_{k}"] for k in ("inner_train", "inner_val", "test")}
        site = {k: c.loc[sp_[k], "SITE_ID"].to_numpy() for k in E}
        y = {k: c.loc[sp_[k], "label"].to_numpy() for k in E}
        cm = pc.combat_fit(E["inner_train"], site["inner_train"], self.cov(sp_["inner_train"]))
        Ec, unseen = {}, {}
        for k in E:
            Ec[k], un = pc.combat_apply(cm, E[k], site[k], self.cov(sp_[k]))
            unseen[k] = int(un.sum())
        rows, mets, site_ba = [], {}, {}
        for meth, EE in (("raw", E), ("combat", Ec)):
            pred, extra, _ = pm.fit_probe(EE["inner_train"], y["inner_train"], EE["inner_val"], y["inner_val"],
                                          [0.001, 0.01, 0.1, 1.0], self.cfg["seed"] + fold)
            for part in ("inner_val", "test"):
                for s, p in zip(sp_[part], pred(EE[part])):
                    rows.append({"SUB_ID": int(s), "SITE_ID": c.at[s, "SITE_ID"], "fold": fold, "part": part,
                                 "label": int(c.at[s, "label"]), "method": meth, "prob": float(p)})
            sc = StandardScaler().fit(EE["inner_train"])
            clf = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(EE["inner_train"]), site["inner_train"])
            site_ba[meth] = float(balanced_accuracy_score(site["test"], clf.predict(sc.transform(EE["test"]))))
        df = pd.DataFrame(rows)
        d = pc.unit_dir(self.rdir, unit); d.mkdir(parents=True, exist_ok=True)
        pc.sp.write_csv_atomic(d / "preds.csv", df)
        t = df[df.part == "test"]
        pc.sp.write_json_atomic(d / "metrics.json", {"test": {m_: pc.metrics(g.label, g.prob) for m_, g in t.groupby("method")},
                                                     "site_ba_test": site_ba, "unseen": unseen,
                                                     "n_sites_test": int(len(set(site["test"])))})
        return {"seconds": 0}

    # ---------------- seleksi, fusion, ensemble ----------------
    def sfcn_sanity_ok(self):
        """Cek kewajaran brain age SFCN (tanpa diagnosis). Gagal -> model sfcn_* dikeluarkan dari seleksi/fusion."""
        try:
            info = json.loads((pc.unit_dir(self.rdir, "cache_sfcn64") / "cache_done.json").read_text(encoding="utf-8"))
            a = pd.read_csv(self.cache_dir("sfcn64") / "brain_age_sfcn.csv")
        except Exception:
            return False
        rule = self.cfg["sfcn_sanity_rule"]
        adult = info.get("sanity_brain_age", {}).get("usia>=18", {})
        return bool(a["brain_age_sfcn"].std() >= rule["min_pred_sd_years"] and abs(adult.get("pearson_r", 0)) >= rule["min_abs_r_adult"])

    def selection(self):
        sel, allc = {}, {}
        sfcn_ok = self.sfcn_sanity_ok()
        for mod, names in self.cfg["modality"].items():
            cand = {}
            for n in names:
                if n.startswith("sfcn") and not sfcn_ok:
                    continue
                us = [f"{n}:{k}" for k in self.splits]
                if all(pc.unit_done_valid(self.rdir, u) for u in us):
                    p = pd.concat([pd.read_csv(pc.unit_dir(self.rdir, u) / "preds.csv") for u in us])
                    v = p[p.part == "inner_val"]
                    cand[n] = pc.metrics(v.label, v.prob)["auc"]
            sel[mod] = {"best": max(cand, key=cand.get) if cand else None, "auc_inner_val": cand}
            allc.update(cand)
        img = {**sel["img25d"]["auc_inner_val"], **sel["img3d"]["auc_inner_val"]}
        sel["img"] = {"best": max(img, key=img.get) if img else None, "auc_inner_val": img}
        sel["top3"] = sorted(allc, key=allc.get, reverse=True)[:self.cfg["ensemble"]["top_k"]]
        sel["sfcn_sanity_ok"] = sfcn_ok
        return sel

    def scen_members(self, scen, sel):
        if scen == "ENS":
            return sel["top3"] if len(sel["top3"]) == self.cfg["ensemble"]["top_k"] else None
        mem = [sel[m]["best"] for m in self.cfg["fusion"]["scenarios"][scen]]
        return None if any(m is None for m in mem) else mem

    def fusion_current(self, unit):
        try:
            m = json.loads((pc.unit_dir(self.rdir, unit) / "metrics.json").read_text(encoding="utf-8"))
            scen = unit.split(":")[0].split("_", 1)[1]
            return m["members"] == self.scen_members(scen, self.selection())
        except Exception:
            return False

    def run_fusion_v2(self, unit, scen, fold):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.preprocessing import StandardScaler
        import pilot_models as pm
        sel = self.selection()
        mem = self.scen_members(scen, sel)
        if mem is None:
            raise RuntimeError(f"SKIP: anggota fusion {scen} belum lengkap 5 fold")
        sp_ = self.splits[fold]
        c = self.cohort.set_index("SUB_ID")
        P = {n: pd.read_csv(pc.unit_dir(self.rdir, f"{n}:{fold}") / "preds.csv").set_index(["part", "SUB_ID"])["prob"]
             for n in mem}
        tabular = all(self.cfg["models"][n]["kind"] in ("pheno", "roi") for n in mem)
        out = {}
        te = sp_["test"]
        inner = {}                                   # probabilitas inner untuk threshold Youden (tidak memakai test)
        if scen == "ENS":
            out["ENS_mean_top3"] = np.mean([P[n].loc["test"].loc[te].to_numpy() for n in mem], 0)
            vi = sp_["inner_val"]
            inner["ENS_mean_top3"] = (vi, np.mean([P[n].loc["inner_val"].loc[vi].to_numpy() for n in mem], 0))
        else:
            T = np.concatenate([sp_["inner_train"], sp_["inner_val"]]) if tabular else sp_["inner_val"]
            part = "inner_oof" if tabular else "inner_val"
            yT = c.loc[T, "label"].to_numpy()
            Xs = np.column_stack([lg(P[n].loc[part].loc[T].to_numpy()) for n in mem])
            Xt = np.column_stack([lg(P[n].loc["test"].loc[te].to_numpy()) for n in mem])
            out["F1_stack_logreg"] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Xs, yT).predict_proba(Xt)[:, 1]
            out["F1_mean_prob"] = np.mean([P[n].loc["test"].loc[te].to_numpy() for n in mem], 0)
            cv = StratifiedKFold(5, shuffle=True, random_state=fold)
            inner["F1_stack_logreg"] = (T, cross_val_predict(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
                                                             Xs, yT, cv=cv, method="predict_proba")[:, 1])
            inner["F1_mean_prob"] = (T, np.mean([P[n].loc[part].loc[T].to_numpy() for n in mem], 0))
            # F2: fitur gabungan -> LogReg. Representasi di-fit pada train (inner_train; atau T untuk tabel saja)
            fit_ids = T if tabular else sp_["inner_train"]
            k = self.cfg["fusion"]["pca_dims"]
            blocks_T, blocks_t = [], []
            pre = pm.TabPre("pheno", self.features).fit(self.pheno.loc[fit_ids])
            blocks_T.append(pre.transform(self.pheno.loc[T])); blocks_t.append(pre.transform(self.pheno.loc[te]))
            roi_m = next((n for n in mem if self.cfg["models"][n]["kind"] == "roi"), None)
            if roi_m:
                R = self.roi_table()
                rp_ = pm.TabPre("roi", list(R.columns), self.cfg["models"][roi_m]["combat"]).fit(
                    R.loc[fit_ids], c.loc[fit_ids, "SITE_ID"].to_numpy(), self.cov(fit_ids))
                tr_ = lambda ids: rp_.transform(R.loc[ids], c.loc[ids, "SITE_ID"].to_numpy(), self.cov(ids))
                pca = PCA(n_components=k, random_state=fold).fit(tr_(fit_ids))
                blocks_T.append(pca.transform(tr_(T))); blocks_t.append(pca.transform(tr_(te)))
            img_m = next((n for n in mem if self.cfg["models"][n]["kind"] not in ("pheno", "roi")), None)
            if img_m:
                e = np.load(pc.unit_dir(self.rdir, f"{img_m}:{fold}") / "emb.npz")
                sc = StandardScaler().fit(e["emb_inner_train"])
                pca = PCA(n_components=min(k, e["emb_inner_train"].shape[1]), random_state=fold).fit(sc.transform(e["emb_inner_train"]))
                blocks_T.append(pca.transform(sc.transform(e["emb_inner_val"])))
                blocks_t.append(pca.transform(sc.transform(e["emb_test"])))
            ZT, Zt = np.hstack(blocks_T), np.hstack(blocks_t)
            out[f"F2_pca{k}_logreg"] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(ZT, yT).predict_proba(Zt)[:, 1]
            inner[f"F2_pca{k}_logreg"] = (T, cross_val_predict(LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000),
                                                               ZT, yT, cv=cv, method="predict_proba")[:, 1])
        rows = [{"SUB_ID": int(s), "SITE_ID": c.at[s, "SITE_ID"], "fold": fold, "part": "test", "label": int(c.at[s, "label"]),
                 "method": mth, "prob": float(p)} for mth, ps in out.items() for s, p in zip(te, ps)]
        rows += [{"SUB_ID": int(s), "SITE_ID": c.at[s, "SITE_ID"], "fold": fold, "part": "inner_cv", "label": int(c.at[s, "label"]),
                  "method": mth, "prob": float(p)} for mth, (ids_, ps) in inner.items() for s, p in zip(ids_, ps)]
        df = pd.DataFrame(rows)
        d = pc.unit_dir(self.rdir, unit); d.mkdir(parents=True, exist_ok=True)
        pc.sp.write_csv_atomic(d / "preds.csv", df)
        pc.sp.write_json_atomic(d / "metrics.json", {"test": {m_: pc.metrics(g.label, g.prob) for m_, g in df[df.part == "test"].groupby("method")},
                                                     "members": mem, "stack_train": "inner_oof" if (scen != "ENS" and tabular)
                                                     else "inner_val", "note": self.cfg["fusion"]["note"]})
        return {"seconds": 0}

    # ---------------- Tabel A otomatis (format sama dengan v1) ----------------
    def maybe_report_A(self, units):
        ph = [u for u in units if u.split(":")[0] in self.cfg["modality"]["pheno"]]
        flag = self.rdir / "tables" / ".tabel_A_dicetak"
        if not ph or flag.exists() or not all(pc.unit_done_valid(self.rdir, u) for u in ph):
            return
        try:
            import make_tables as mt
            import make_tables_v2 as mt2
            mt2.build(self.rdir, self.subset)
            txt = mt.console_A(self.rdir)
            print(txt, flush=True)
            (self.rdir / "tables" / "tabel_A_layar.txt").write_text(txt, encoding="utf-8")
            where = mt.mirror(self.rdir, self.rdir.name) if not self.args.results_dir else None
            flag.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")
            LOG.info("Tabel A disimpan%s | lanjut ke ROI & citra.", f" | salinan: {where}" if where else "")
        except Exception as e:
            LOG.warning("Tabel A gagal dibuat (training tetap lanjut): %s", e)


if __name__ == "__main__":
    sys.exit(rp.main(cls=RunnerV2))
