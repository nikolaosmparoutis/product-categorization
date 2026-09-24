"""Experiment 08 — was the tuning worth it? The one and only look at test.

Compares the hyperparameters that came from experience against the ones Optuna
found on validation. Both are trained on the same train split and measured on
the same test, with identical calibration.

Test has been used for no decision. This is the final measurement, not a
selection criterion: the parameters are already chosen whatever it shows.

What it does, in order:
1. Reads the Optuna parameters from docs/optuna_best_params.json.
2. Trains two models, identical in everything but the hyperparameters.
3. Calibrates both with Platt on the same set.
4. Measures on test and prints the gap next to the measured noise of 0.004.
5. Writes a comparison csv.
"""

import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from common import RANDOM_STATE, TFIDF_SETTINGS, TARGET_COLUMN, load_and_split_4, metrics_row

NOISE = 0.004

PARAMS_BASELINE = dict(
    n_estimators=200, learning_rate=0.05, num_leaves=63, min_child_samples=15,
    subsample=0.75, reg_lambda=0.1, max_bin=60,
)
PARAMS_OPTUNA = json.load(open("docs/optuna_best_params.json"))["params"]

sets = load_and_split_4()
y_true = sets["test"][TARGET_COLUMN].to_numpy()


def build(params: dict) -> Pipeline:
    """Same pipeline; only the LightGBM hyperparameters change."""
    return Pipeline([
        ("features", ColumnTransformer([
            ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"]),
            ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
            ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
            ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
        ])),
        ("lgbm", LGBMClassifier(
            class_weight="balanced", subsample_freq=1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params,
        )),
    ])


results = []
for label, params in [("baseline", PARAMS_BASELINE), ("optuna", PARAMS_OPTUNA)]:
    model = build(params)
    model.fit(sets["train"], sets["train"][TARGET_COLUMN], lgbm__categorical_feature=[0])

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(sets["calibration"], sets["calibration"][TARGET_COLUMN])

    row = metrics_row(label, calibrated.predict_proba(sets["test"]), calibrated.classes_, y_true, duration=0)
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

table = pd.DataFrame(results)
difference = table.loc[1, "macro_f1"] - table.loc[0, "macro_f1"]

print("\nTELIKI SYGKRISI STO TEST")
print(table[["experiment", "macro_f1", "mcc", "accuracy", "auto_share", "storefront_errors"]].to_string(index=False))
print(f"\ndiafora macro F1: {difference:+.4f}  |  thoryvos apo to CV: {NOISE}")
print("SYMPERASMA:", "to tuning edose pragmatiko kerdos" if difference > NOISE else
      "to tuning DEN edose kerdos pera apo ton thoryvo")

table.to_csv("docs/experiment_08_final_comparison.csv", index=False)
