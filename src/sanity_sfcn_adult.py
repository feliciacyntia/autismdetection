"""Sanity check SFCN (ulang): brain age vs AGE_AT_SCAN hanya subjek >= 18 tahun (tanpa label diagnosis).
Varian 'pipeline' = sfcn_input + BN eval (jalur yang dipakai v2). Varian 'bn_input' (diagnostik) = BatchNorm memakai
statistik volume input sendiri, untuk menguji apakah running stats checkpoint menjadi penyebab softmax rata.
Aturan keputusan (CONFIG_V2['sfcn_sanity_rule']) dievaluasi hanya pada varian pipeline kecuali ada bug terbukti."""
import json
import time

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

import pilot_common as pc
import pilot_models as pm

FORBIDDEN = ("DX_GROUP", "ADOS", "ADI_R", "SRS", "SCQ", "AQ", "DSM_IV_TR", "CURRENT_MED_STATUS")


def main():
    cfg = pc.CONFIG_V2
    rdir = pc.results_dir("pilot200", "v2")
    out = rdir / "sanity"; out.mkdir(exist_ok=True)
    coh, _ = pc.load_cohort("pilot200")
    ph, _ = pc.load_pheno(coh)
    age = ph["AGE_AT_SCAN"]
    assert not any(c.startswith(FORBIDDEN) for c in [age.name])
    ids = np.array([s for s in coh["SUB_ID"] if age.loc[s] >= 18])
    ages = age.loc[ids].to_numpy(float)
    net = pm.sfcn_pretrained(cfg["sfcn"]).to(pm.DEV).eval()
    bins = torch.arange(40, device=pm.DEV) + 42.5
    pred = {"pipeline": [], "bn_input": [], "pmax_pipeline": [], "pmax_bn_input": []}
    t0 = time.time()
    for i in range(0, len(ids), 4):
        V, M = pc.load_volumes(ids[i:i + 4])
        for v, m in zip(V, M):
            x = pm.sfcn_input(torch.from_numpy(v)[None].to(pm.DEV), torch.from_numpy(m)[None].to(pm.DEV))
            for var in ("pipeline", "bn_input"):
                net.eval()
                if var == "bn_input":
                    for mod in net.modules():
                        if isinstance(mod, torch.nn.BatchNorm3d):
                            mod.train(); mod.track_running_stats = False
                with torch.no_grad():
                    p = torch.softmax(net.classifier(net.feature_extractor(x)).view(-1), 0)
                pred[var].append(float((p * bins).sum())); pred["pmax_" + var].append(float(p.max()))
                for mod in net.modules():
                    if isinstance(mod, torch.nn.BatchNorm3d):
                        mod.track_running_stats = True
    rule = cfg["sfcn_sanity_rule"]
    res = {"tanggal": time.strftime("%Y-%m-%d %H:%M"), "n": int(len(ids)), "usia_min": float(ages.min()),
           "usia_median": float(np.median(ages)), "usia_max": float(ages.max()), "rentang_model": [42, 82],
           "n_usia_ge_42": int((ages >= 42).sum()), "aturan": rule, "menit": (time.time() - t0) / 60}
    for var in ("pipeline", "bn_input"):
        pr = np.array(pred[var])
        res[var] = {"pred_mean": float(pr.mean()), "pred_sd": float(pr.std()), "pmax_median": float(np.median(pred["pmax_" + var])),
                    "pearson_r": float(pearsonr(pr, ages)[0]), "pearson_p": float(pearsonr(pr, ages)[1]),
                    "spearman_r": float(spearmanr(pr, ages)[0]), "mae": float(np.abs(pr - ages).mean())}
    print(json.dumps(res, indent=1))
    p = out / "sfcn_adult_check.json"
    tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(res, indent=1), encoding="utf-8"); tmp.replace(p)


if __name__ == "__main__":
    main()
