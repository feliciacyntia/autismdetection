"""Pipeline preprocessing fenotipik ABIDE I (dipakai notebook 05b dan tahap pemodelan).

Modul ini hanya MEMBANGUN pipeline yang belum di-fit. Fit selalu dilakukan
pemanggil pada data latih satu fold; tidak ada transformasi global di sini.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ("AGE_AT_SCAN", "FIQ")              # imputasi median (+ scaling untuk linear/NN)
CATEGORICAL_FEATURES = ("SEX", "HANDEDNESS_CATEGORY")  # imputasi kategori lalu one-hot
BINARY_FEATURES = ("FIQ_missing",)                     # 0/1 tanpa missing, diteruskan apa adanya
ALLOWED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

FORBIDDEN_EXACT = {
    "DX_GROUP", "label", "SUB_ID", "SITE_ID", "group_id", "smri_path", "qc_decision", "qc_reason",
    "loso_site", "loso_group_conflict", "loso_eligible", "strata_fallback", "image_integrity_flag",
}
# Kolom fold, QC, hash/flag audit, dan instrumen diagnostik/klinis
FORBIDDEN_PATTERN = re.compile(
    r"fold|^qc_|^flag_|^outlier_|sha256|^hash|^DX_|DSM|^ADOS|^ADI|^SRS|^SCQ|^AQ_|^AQ$|"
    r"VINELAND|^WISC|COMORBID|MEDICATION|CURRENT_MED|OFF_STIMULANTS",
    re.IGNORECASE,
)


def forbidden_columns(columns):
    """Kembalikan kolom yang tidak boleh menjadi fitur."""
    return [c for c in columns if c in FORBIDDEN_EXACT or FORBIDDEN_PATTERN.search(str(c))]


def assert_no_forbidden(columns):
    bad = forbidden_columns(columns)
    assert not bad, f"Kolom terlarang di X: {bad}"


def select_features(df, config):
    """Pilih fitur sesuai config["features"], urutan tetap: numerik, kategori, biner.

    Hanya fitur di ALLOWED_FEATURES yang diterima; kolom lain (termasuk seluruh
    kolom numerik lain) tidak pernah ikut otomatis.
    """
    requested = list(config["features"])
    unknown = [f for f in requested if f not in ALLOWED_FEATURES]
    assert not unknown, f"Fitur di luar daftar yang diizinkan: {unknown}"
    assert_no_forbidden(requested)
    missing = [f for f in requested if f not in df.columns]
    assert not missing, f"Fitur tidak ada di data: {missing}"
    ordered = [f for f in ALLOWED_FEATURES if f in requested]
    return df[ordered].copy()


def split_feature_types(features):
    features = list(features)
    return ([f for f in features if f in NUMERIC_FEATURES],
            [f for f in features if f in CATEGORICAL_FEATURES],
            [f for f in features if f in BINARY_FEATURES])


def _category_value(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)) or (isinstance(v, (float, np.floating)) and float(v).is_integer()):
        return str(int(v))
    return str(v).strip()


def _as_category_str(X):
    """Ubah nilai kategori ke string (1 / 1.0 -> "1"), NaN tetap NaN, agar one-hot konsisten."""
    X = pd.DataFrame(X).copy()
    for c in X.columns:
        X[c] = X[c].map(_category_value).astype(object)
    return X


def _categorical_pipe(missing_strategy):
    if missing_strategy == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent")
    elif missing_strategy == "missing_category":
        imputer = SimpleImputer(strategy="constant", fill_value="Missing")
    else:
        raise ValueError(f"missing_strategy tidak dikenal: {missing_strategy}")
    return Pipeline([
        ("to_str", FunctionTransformer(_as_category_str, feature_names_out="one-to-one")),
        ("imputer", imputer),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def _column_transformer(features, numeric_pipe, cat_missing_strategy):
    num, cat, binary = split_feature_types(features)
    assert len(num) + len(cat) + len(binary) == len(list(features)), "Ada fitur yang tidak terpetakan"
    transformers = []
    if num:
        transformers.append(("num", numeric_pipe, num))
    if cat:
        transformers.append(("cat", _categorical_pipe(cat_missing_strategy), cat))
    if binary:
        transformers.append(("bin", "passthrough", binary))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def make_pipeline_tree(features, cat_missing_strategy="most_frequent"):
    """Untuk model pohon (LightGBM, dsb.): median numerik, imputasi + one-hot kategori, tanpa scaling."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return Pipeline([("prep", _column_transformer(features, numeric, cat_missing_strategy))])


def make_pipeline_linear_nn(features, cat_missing_strategy="most_frequent"):
    """Untuk LogReg/NN: median + StandardScaler numerik, imputasi + one-hot kategori."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    return Pipeline([("prep", _column_transformer(features, numeric, cat_missing_strategy))])


def get_numeric_step(pipeline, step):
    """Ambil step ('imputer'/'scaler') dari cabang numerik pipeline yang sudah di-fit."""
    return pipeline.named_steps["prep"].named_transformers_["num"].named_steps[step]
