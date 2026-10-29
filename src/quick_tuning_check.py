"""Uji kilat (diagnostik, pilot): apakah tuning hyperparameter NESTED menaikkan AUC dibanding parameter tetap?
Tuning hanya di data latih tiap fold (inner 3-fold, skor AUC); test fold dipakai sekali untuk evaluasi. Tidak mengubah config terkunci.
    .venv-model\\Scripts\\python.exe src\\quick_tuning_check.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot_common as pc  # noqa: E402
import pilot_models as pm  # noqa: E402

warnings.filterwarnings("ignore")
RD = pc.results_dir("pilot200", "v2")
SPL = {int(k): {p: np.asarray(v) for p, v in s.items()} for k, s in json.loads((RD / "splits.json").read_text(encoding="utf-8")).items()}
cohort, _ = pc.load_cohort("pilot200")
pheno, _ = pc.load_pheno(cohort)
FEAT = pc.missing_report(cohort, pheno).query("used").feature.tolist()
ROI = pd.read_csv(RD / "cache" / "roi" / "roi_features.csv").set_index("SUB_ID").loc[cohort.SUB_ID]
lab = cohort.set_index("SUB_ID")["label"]
DATA = {"Fenotipik": pheno[FEAT], "ROI (tanpa ComBat)": ROI}


def pre(kind):
    return pm.pheno_preprocessor(FEAT) if kind == "Fenotipik" else Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])


def logreg(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=5000)


def xgb():
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, eval_metric="logloss",
                         random_state=42, n_jobs=4, tree_method="hist")


def run(kind, make, search=None):
    P, Y, best = [], [], []
    for k, s in SPL.items():
        T = np.concatenate([s["inner_train"], s["inner_val"]])
        X, y = DATA[kind], lab
        pipe = Pipeline([("pre", pre(kind)), ("m", make())])
        if search is not None:
            pipe = search(pipe, StratifiedKFold(3, shuffle=True, random_state=k))
        pipe.fit(X.loc[T], y.loc[T].to_numpy())
        if search is not None:
            best.append({kk.replace("m__", ""): (round(v, 4) if isinstance(v, float) else v) for kk, v in pipe.best_params_.items()})
        P.append(pipe.predict_proba(X.loc[s["test"]])[:, 1]); Y.append(y.loc[s["test"]].to_numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    yh = (P >= 0.5).astype(int)
    ba = 0.5 * (((yh == 1) & (Y == 1)).sum() / (Y == 1).sum() + ((yh == 0) & (Y == 0)).sum() / (Y == 0).sum())
    ci = pc.bootstrap_ci(Y, P)
    return {"AUC": f"{roc_auc_score(Y, P):.3f} [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]",
            "BA": f"{ba:.3f} [{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]", "param terpilih (per fold)": best}


grid_C = lambda p, cv: GridSearchCV(p, {"m__C": np.logspace(-4, 1, 16)}, scoring="roc_auc", cv=cv, n_jobs=4)
rand_xgb = lambda p, cv: RandomizedSearchCV(p, {"m__max_depth": randint(2, 6), "m__learning_rate": loguniform(0.01, 0.3),
                                                "m__n_estimators": randint(50, 500), "m__subsample": uniform(0.6, 0.4),
                                                "m__min_child_weight": randint(1, 10), "m__reg_lambda": loguniform(0.1, 10)},
                                            n_iter=25, scoring="roc_auc", cv=cv, random_state=42, n_jobs=4)
t0 = time.time()
rows = []
for kind in DATA:
    rows.append({"fitur": kind, "model": "LogReg", "versi": "tetap (C = 1)", **run(kind, logreg)})
    rows.append({"fitur": kind, "model": "LogReg", "versi": "tuning nested (C, 16 nilai)", **run(kind, logreg, grid_C)})
    rows.append({"fitur": kind, "model": "XGBoost", "versi": "tetap", **run(kind, xgb)})
    rows.append({"fitur": kind, "model": "XGBoost", "versi": "tuning nested (random 25)", **run(kind, xgb, rand_xgb)})
    print(f"{kind} selesai ({time.time() - t0:.0f} s)", flush=True)
df = pd.DataFrame(rows)
out = RD / "report" / "tables" / "08b_quick_tuning_check.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(df[["fitur", "model", "versi", "AUC", "BA"]].to_string(index=False))
print("tersimpan:", out)
