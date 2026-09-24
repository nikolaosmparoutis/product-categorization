"""Experiment 03 — feature ablations, same split, calibration and hyperparameters.

v1 reference · v4 without brand (only 1.1% of importance) · v5 one joined text field.

What it does, in order:
1. Defines three feature blocks: full, no brand, joined text.
2. Trains and calibrates each on the same split with the same hyperparameters.
3. Writes a csv, so the cost of removing each design decision is visible.
"""

import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from common import (
    RANDOM_STATE,
    TFIDF_SETTINGS,
    TEXT_COLUMNS,
    train_and_evaluate,
    load_and_split,
)

BLOK_BRAND = ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"])


def join_fields(data: pd.DataFrame) -> pd.Series:
    """name + description + features as one text: the baseline the EDA rejected."""
    return data[TEXT_COLUMNS].agg(" ".join, axis=1)


transformers = {
    "v1_full": ColumnTransformer([
        BLOK_BRAND,
        ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
        ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
        ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
    ]),
    "v4_no_brand": ColumnTransformer([
        ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
        ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
        ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
    ]),
    "v5_joined_text": Pipeline([
        ("union", FunctionTransformer(join_fields)),
        ("tfidf", TfidfVectorizer(**TFIDF_SETTINGS)),
    ]),
}


def build_model(transformer) -> Pipeline:
    """Same LightGBM as v1; only the feature block changes."""
    return Pipeline([
        ("features", transformer),
        ("lgbm", LGBMClassifier(
            class_weight="balanced", n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=15, subsample=0.75, subsample_freq=1, reg_lambda=0.1, max_bin=60,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ])


sets = load_and_split()
results = []
for label, transformer in transformers.items():
    fit_params = {"lgbm__categorical_feature": [0]} if label == "v1_full" else {}
    result = train_and_evaluate(label, build_model(transformer), sets, **fit_params)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    results.append(result)

pd.DataFrame(results).to_csv("docs/experiment_03_ablations.csv", index=False)
print(pd.DataFrame(results).to_string())
