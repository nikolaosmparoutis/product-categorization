"""Experiment 04 — isotonic or sigmoid calibration?

Suspicion: isotonic overfits the small classes, which hold only 22 to 39
products in the calibration set. The clue was that macro F1 dropped after
calibration while accuracy stayed flat, meaning the damage was confined to the
small classes.

The model is trained ONCE and calibrated three ways, so the comparison carries
no training randomness.

What it does, in order:
1. Trains the base model once.
2. Calibrates three ways: none, isotonic, sigmoid.
3. Prints a reliability table per probability band and F1 per class.
4. Writes a comparison csv.
"""

import json

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from common import (
    RANDOM_STATE,
    TFIDF_SETTINGS,
    TARGET_COLUMN,
    load_and_split,
    metrics_row,
)

ZONES = [0.0, 0.5, 0.9, 0.95, 0.99, 1.01]


def reliability_table(probabilities, predictions, y_true) -> pd.DataFrame:
    """Compare the claimed probability against the observed accuracy, per band.

    This is the criterion that matters: "0.99" should really mean 99%.
    The deviation column is the size of the lie in each band.
    """
    p_max = probabilities.max(axis=1)
    band = pd.cut(p_max, ZONES, right=True, include_lowest=True)
    per_band = pd.DataFrame({"band": band, "p_max": p_max, "correct_mask": predictions == y_true})
    summary = per_band.groupby("band", observed=True).agg(
        count=("correct_mask", "size"), mean_probability=("p_max", "mean"), actual=("correct_mask", "mean")
    )
    summary["apoklisi"] = (summary["mean_probability"] - summary["actual"]).abs().round(4)
    return summary.round(4)


sets = load_and_split()
y_true = sets["test"][TARGET_COLUMN].to_numpy()

transformer = ColumnTransformer(
    [
        ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"]),
        ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
        ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
        ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
    ]
)
model = Pipeline(
    [
        ("features", transformer),
        ("lgbm", LGBMClassifier(
            class_weight="balanced", n_estimators=200, learning_rate=0.05, num_leaves=63,
            min_child_samples=15, subsample=0.75, subsample_freq=1, reg_lambda=0.1, max_bin=60,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )),
    ]
)

print("training_data mias foras...", flush=True)
model.fit(sets["train"], sets["train"][TARGET_COLUMN], lgbm__categorical_feature=[0])

variants = {"no_calibration": None, "isotonic": "isotonic", "sigmoid": "sigmoid"}
results = []
for label, method_name in variants.items():
    if method_name is None:
        probabilities, classes_list = model.predict_proba(sets["test"]), model.classes_
    else:
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method=method_name)
        calibrated.fit(sets["calibration"], sets["calibration"][TARGET_COLUMN])
        probabilities, classes_list = calibrated.predict_proba(sets["test"]), calibrated.classes_

    result = metrics_row(label, probabilities, classes_list, y_true, duration=0)
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(reliability_table(probabilities, classes_list[probabilities.argmax(axis=1)], y_true).to_string(), flush=True)

    f1_per_class = pd.Series(
        f1_score(y_true, classes_list[probabilities.argmax(axis=1)], average=None, labels=classes_list),
        index=classes_list,
    ).round(4)
    print("F1 ana cls:", f1_per_class.to_dict(), "\n", flush=True)

pd.DataFrame(results).to_csv("docs/experiment_04_calibration.csv", index=False)
print(pd.DataFrame(results).to_string())
