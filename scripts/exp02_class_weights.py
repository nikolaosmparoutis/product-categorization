"""Experiment 02 — does a lower weight on class 5 reduce the 1049-as-5 errors?

Three variants sharing the same split and the same calibration; only
class_weight changes.

What it does, in order:
1. Computes the weights class_weight="balanced" would produce.
2. Builds three variants, lowering ONLY the weight of the majority class 5.
3. Trains and calibrates each one on the same split.
4. Writes a comparison csv, focused on the "truly 1049, predicted 5" errors.
"""

import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from common import (
    MAJORITY_CLASS,
    RANDOM_STATE,
    TFIDF_SETTINGS,
    train_and_evaluate,
    load_and_split,
    balanced_weights,
)


def build_model(class_weight) -> Pipeline:
    """The v1 pipeline (brand categorical plus three TF-IDF blocks), with class weights as a parameter."""
    transformer = ColumnTransformer(
        [
            ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"]),
            ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
            ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
            ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
        ]
    )
    return Pipeline(
        [
            ("features", transformer),
            ("lgbm", LGBMClassifier(
                class_weight=class_weight, n_estimators=200, learning_rate=0.05, num_leaves=63,
                min_child_samples=15, subsample=0.75, subsample_freq=1, reg_lambda=0.1, max_bin=60,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            )),
        ]
    )


sets = load_and_split()
weights = balanced_weights(sets["train"])
variants = {
    "v1_balanced": "balanced",
    "v2_majority_x0.5": {**weights, MAJORITY_CLASS: weights[MAJORITY_CLASS] * 0.5},
    "v3_majority_x0.25": {**weights, MAJORITY_CLASS: weights[MAJORITY_CLASS] * 0.25},
}

results = []
for label, class_weight in variants.items():
    result = train_and_evaluate(
        label, build_model(class_weight), sets, lgbm__categorical_feature=[0]
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    results.append(result)

pd.DataFrame(results).to_csv("docs/experiment_02_class_weights.csv", index=False)
print(pd.DataFrame(results).to_string())
