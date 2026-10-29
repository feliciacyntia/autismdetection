"""06 — Tabel & gambar PILOT v2 (bisa dijalankan kapan saja; unit belum selesai = "belum ada").

    .venv-model\\Scripts\\python.exe src\\make_tables_v2.py --subset pilot200
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402
import make_tables as mt  # noqa: E402

V1_MAP = {"logreg": "logreg", "xgb": "xgb", "mlp_tab": "mlp_tab", "tabm": "tabm", "effnet_b0": "effnet_b0",
          "resnet18": "resnet18", "convnextv2_t": "convnextv2_t", "sfcn_lp": "sfcn", "sfcn_head": "sfcn",
          "dinov2_lp": "dinov2_s", "dinov2_head": "dinov2_s"}


def infos(rdir, n, folds):
    out = []
    for k in folds:
        try:
            out.append(json.loads((pc.unit_dir(rdir, f"{n}:{k}") / "metrics.json").read_text(encoding="utf-8"))["info"])
        except Exception:
            pass
    return out


def row_for(rdir, n, folds, label=None):
    p = mt.load_preds(rdir, n, folds)
    if p is None:
        return None
    row, pooled, _ = mt.summarize(p)
    v = p[p.part == "inner_val"]
    return {"model": label or n, "AUC inner val": round(pc.metrics(v.label, v.prob)["auc"], 3), **row}


def build(rdir, subset):
    cfgj = json.loads((rdir / "config.json").read_text(encoding="utf-8"))
    cfg = cfgj["config"]
    splits = json.loads((rdir / "splits.json").read_text(encoding="utf-8"))
    folds = sorted(int(k) for k in splits)
    tdir, fdir = rdir / "tables", rdir / "figures"
    tdir.mkdir(exist_ok=True); fdir.mkdir(exist_ok=True)
    status = pc.read_status(rdir)
    v1dir = pc.results_dir(subset, "v1")
    md = [f"# Hasil PILOT v2 {subset} (config_hash {cfgj['config_hash']})", "",
          f"Status unit: {status['status'].value_counts().to_dict()}", ""]

    # ---- seleksi (inner val pooled) & kriteria "sudah belajar" ----
    sel_rows, best = [], {}
    for mod, names in cfg["modality"].items():
        cand = {}
        for n in names:
            p = mt.load_preds(rdir, n, folds)
            inf = infos(rdir, n, folds)
            learned = [i.get("learned") for i in inf if "learned" in i]
            r = {"modalitas": mod, "model": n, "kelompok": cfg["models"][n]["group"]}
            if p is None:
                r["AUC inner val (pooled)"] = "belum ada"
            else:
                v = p[p.part == "inner_val"]
                cand[n] = pc.metrics(v.label, v.prob)["auc"]
                r["AUC inner val (pooled)"] = round(cand[n], 3)
            if learned:
                r["sudah belajar (fold)"] = f"{sum(learned)}/{len(learned)}"
            sel_rows.append(r)
        best[mod] = sorted(cand, key=cand.get, reverse=True)
    sel = pd.DataFrame(sel_rows)

    # ---- Tabel A (format v1) ----
    tA = mt.table_A(rdir, cfg, folds, splits, best.get("pheno", []))
    tA.to_csv(tdir / "tabel_A_fenotipik.csv", index=False)

    # ---- v1 vs v2 ----
    rows = []
    for n2, n1 in V1_MAP.items():
        r1 = row_for(v1dir, n1, folds, n1) if v1dir.exists() else None
        r2 = row_for(rdir, n2, folds, n2)
        for ver, r in (("v1", r1), ("v2", r2)):
            if r is not None and not (ver == "v1" and any(x["versi"] == "v1" and x["model"] == n1 for x in rows)):
                rows.append({"versi": ver, **r})
    tV = pd.DataFrame(rows)
    tV.to_csv(tdir / "tabel_v1_vs_v2.csv", index=False)

    # ---- citra v2 detail (pretrained vs dari nol) ----
    rowsB = []
    for n in cfg["modality"]["img25d"] + cfg["modality"]["img3d"]:
        r = row_for(rdir, n, folds)
        inf = infos(rdir, n, folds)
        if r is None:
            rowsB.append({"model": n, "kelompok": cfg["models"][n]["group"], "status": "belum ada"}); continue
        ex = {}
        if inf and "params_total" in inf[0]:
            ex = {"params latih/total": f"{inf[0].get('params_trainable')}/{inf[0]['params_total']}",
                  "VRAM puncak (MiB)": round(max(i["vram_peak_mib"] for i in inf)),
                  "s/epoch": round(np.mean([i["s_per_epoch"] for i in inf]), 1),
                  "waktu (menit)": round(sum(i["seconds"] for i in inf) / 60, 1),
                  "epoch terbaik (median)": int(np.median([i["best_epoch"] for i in inf])),
                  "sudah belajar": f"{sum(i['learned'] for i in inf)}/{len(inf)}"}
        elif inf and "C" in inf[0]:
            ex = {"C terpilih (per fold)": ",".join(str(i["C"]) for i in inf)}
        rowsB.append({"kelompok": cfg["models"][n]["group"], **r, **ex})
    tB = pd.DataFrame(rowsB); tB.to_csv(tdir / "tabel_B_citra_v2.csv", index=False)

    # ---- ROI dengan/tanpa ComBat ----
    rowsR = []
    for n in cfg["modality"]["roi"]:
        r = row_for(rdir, n, folds)
        hp = cfg["models"][n]
        rowsR.append({"model": n, "basis": hp["base"], "ComBat": "ya" if hp["combat"] else "tidak",
                      **(r or {"status": "belum ada"})})
    tR = pd.DataFrame(rowsR); tR.to_csv(tdir / "tabel_ROI.csv", index=False)

    # ---- fusion & ensemble ----
    rowsF = []
    for scen in list(cfg["fusion"]["scenarios"]) + ["ENS"]:
        try:
            p = pd.concat([pd.read_csv(pc.unit_dir(rdir, f"fusion_{scen}:{k}") / "preds.csv") for k in folds])
            meta = json.loads((pc.unit_dir(rdir, f"fusion_{scen}:{folds[0]}") / "metrics.json").read_text(encoding="utf-8"))
        except Exception:
            rowsF.append({"skenario": scen, "metode": "belum ada"}); continue
        for mth, g in p[p.part == "test"].groupby("method"):          # baris inner_cv (threshold Youden) tidak ikut
            row, _, _ = mt.summarize(g)
            rowsF.append({"skenario": scen, "anggota": "+".join(meta["members"]), "metode": mth,
                          "stacking dilatih pada": meta.get("stack_train"), **row})
    tF = pd.DataFrame(rowsF); tF.to_csv(tdir / "tabel_fusion_ensemble.csv", index=False)

    # ---- site learning sebelum/sesudah ComBat (embedding) ----
    rowsE = []
    for n in cfg["modality"]["img25d"] + cfg["modality"]["img3d"]:
        ms = []
        for k in folds:
            try:
                ms.append(json.loads((pc.unit_dir(rdir, f"embcb_{n}:{k}") / "metrics.json").read_text(encoding="utf-8")))
            except Exception:
                pass
        if len(ms) < len(folds):
            continue
        p = pd.concat([pd.read_csv(pc.unit_dir(rdir, f"embcb_{n}:{k}") / "preds.csv") for k in folds])
        t = p[p.part == "test"]
        r = {"embedding": n, "BA situs raw (mean fold)": round(np.mean([m["site_ba_test"]["raw"] for m in ms]), 3),
             "BA situs ComBat (mean fold)": round(np.mean([m["site_ba_test"]["combat"] for m in ms]), 3),
             "acak (1/n situs)": round(1 / 20, 3),
             "subjek test situs tak terlihat (total)": int(sum(m["unseen"]["test"] for m in ms))}
        for meth in ("raw", "combat"):
            g = t[t.method == meth]
            mm, ci = pc.metrics(g.label, g.prob), pc.bootstrap_ci(g.label.to_numpy(), g.prob.to_numpy())
            r[f"AUC probe {meth} [95% CI]"] = f"{mm['auc']:.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]"
            r[f"BA probe {meth}"] = round(mm["balanced_accuracy"], 3)
        rowsE.append(r)
    tE = pd.DataFrame(rowsE); tE.to_csv(tdir / "tabel_site_learning_combat.csv", index=False)

    # ---- cek kewajaran SFCN ----
    san = None
    try:
        san = json.loads((pc.unit_dir(rdir, "cache_sfcn64") / "cache_done.json").read_text(encoding="utf-8")).get("sanity_brain_age")
    except Exception:
        pass

    # ---- kurva loss ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for n in cfg["modality"]["img25d"] + cfg["modality"]["img3d"]:
        inf = infos(rdir, n, folds)
        if not inf or "history" not in inf[0]:
            continue
        fig, axes = plt.subplots(1, len(inf), figsize=(3.2 * len(inf), 3), sharey=True)
        for ax, k, i in zip(np.atleast_1d(axes), folds, inf):
            h = pd.DataFrame(i["history"])
            ax.plot(h.epoch, h.train_loss, label="train"); ax.plot(h.epoch, h.val_loss, label="inner val", alpha=0.6)
            ax.plot(h.epoch, h.val_crit, label="val halus"); ax.axhline(i["const_val_loss"], ls="--", c="gray", lw=0.8,
                                                                          label="konstan")
            ax.axvline(i["best_epoch"], ls=":", c="k", lw=0.8)
            ax.set_title(f"fold {k} | belajar={i['learned']}", fontsize=8); ax.set_xlabel("epoch")
        np.atleast_1d(axes)[0].set_ylabel("BCE berbobot"); np.atleast_1d(axes)[0].legend(fontsize=6)
        fig.suptitle(f"{n}: kurva loss train vs inner val", fontsize=9); fig.tight_layout()
        fig.savefig(fdir / f"loss_{n}.png", dpi=110); plt.close(fig)

    md += ["## Seleksi (AUC pooled inner val lima fold; bukan skor test) & kriteria sudah belajar",
           f"Kriteria: {cfg['learned_criterion']}", "", mt.md_table(sel),
           "## Tabel A — fenotipik", mt.md_table(tA[[c for c in tA.columns if not c.startswith('_')]]),
           "\n".join(mt.leak_warnings(tA)) or f"Tidak ada model dengan BA pooled > {mt.LEAK_BA}.", "",
           "## Tabel v1 vs v2 (test per fold mean ± SD; pooled + 95% CI)", mt.md_table(tV),
           "## Citra v2: pretrained vs dari nol", mt.md_table(tB),
           "## Fitur ROI (Harvard-Oxford, mean & SD intensitas) dengan/tanpa ComBat", mt.md_table(tR),
           "## Fusion & ensemble", mt.md_table(tF),
           "## Site learning sebelum/sesudah ComBat (embedding citra)", mt.md_table(tE),
           "## Cek kewajaran SFCN (brain age vs AGE_AT_SCAN; tanpa label diagnosis)",
           "```\n" + json.dumps(san, indent=1) + "\n```" if san else "_belum ada_", "",
           "## Catatan", "- Pilot n=200: CI lebar; selisih kecil belum tentu bermakna.",
           f"- ROI: {cfg['roi']['note']}.", f"- Fusion: {cfg['fusion']['note']}.",
           f"- ComBat: {cfg['combat']['unseen_site']}."]
    (rdir / "ringkasan_tabel.md").write_text("\n".join(md), encoding="utf-8")
    print("Tabel v2 ditulis ke", tdir)
    print("Status:", status["status"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["pilot200", "all"], default="pilot200")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--no-mirror", action="store_true")
    a = ap.parse_args()
    rdir = Path(a.results_dir) if a.results_dir else pc.results_dir(a.subset, "v2")
    build(rdir, a.subset)
    if not a.no_mirror and not a.results_dir:
        print("Disalin ke", mt.mirror(rdir, rdir.name))


if __name__ == "__main__":
    main()
