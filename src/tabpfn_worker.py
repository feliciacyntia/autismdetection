"""Worker TabPFN v2 (dijalankan dengan .venv-tabpfn): baca job .npz (Xtr_i, ytr_i, Xte_i), tulis prob. ke .npz output.

    .venv-tabpfn\\Scripts\\python.exe src\\tabpfn_worker.py in.npz out.npz n_estimators random_state
"""
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def main():
    src, dst, n_est, rs = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    import torch
    import tabpfn
    from tabpfn import TabPFNClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(src)
    out = {}
    for i in range(int(d["n_jobs"])):
        m = TabPFNClassifier(n_estimators=n_est, device=dev, random_state=rs, ignore_pretraining_limits=dev == "cpu")
        m.fit(d[f"Xtr_{i}"], d[f"ytr_{i}"])
        out[f"p_{i}"] = m.predict_proba(d[f"Xte_{i}"])[:, 1]
    tmp = dst + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, dst)
    print(json.dumps({"tabpfn": tabpfn.__version__, "torch": torch.__version__, "device": dev}))


if __name__ == "__main__":
    main()
