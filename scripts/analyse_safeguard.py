"""Recompute the safeguard tables for the saved model.

What it does, in order:
1. Loads the production model; trains nothing.
2. Predicts on test and keeps the probabilities.
3. Computes threshold sensitivity across four assumed costs of an error.
4. Computes the automatic share per category, which shows where human time goes.
5. Writes two csv files into docs/.
"""

import joblib
import pandas as pd

from common import COST_ERROR, COST_REVIEW, TARGET_COLUMN, load_and_split

ERROR_COSTS_EUR = [0.5, 1.0, 2.0, 5.0]

sets = load_and_split()
y_true = sets["test"][TARGET_COLUMN].to_numpy()

model = joblib.load("model_storage/calibrated_model.joblib")
probabilities = model.predict_proba(sets["test"])
predictions = model.classes_[probabilities.argmax(axis=1)]
p_max = probabilities.max(axis=1)
correct_mask = predictions == y_true

rows = []
for cost_error in ERROR_COSTS_EUR:
    threshold = 1 - (COST_REVIEW / cost_error)
    is_auto = p_max >= threshold
    rows.append({
        "error_cost_eur": cost_error,
        "threshold": round(threshold, 4),
        "auto_share": round(is_auto.mean(), 4),
        "review_share": round(1 - is_auto.mean(), 4),
        "person_hours": round((~is_auto).sum() * 20 / 3600, 1),
        "precision_on_auto": round(correct_mask[is_auto].mean(), 5),
        "storefront_errors": int((~correct_mask[is_auto]).sum()),
    })
sensitivity = pd.DataFrame(rows)
print("EUAISTHISIA KATOFLIOU")
print(sensitivity.to_string(index=False))

production_threshold = 1 - (COST_REVIEW / COST_ERROR)
auto_production = p_max >= production_threshold
ana_klasi = pd.DataFrame({"cls": y_true, "auto": auto_production}).groupby("cls")["auto"].agg(
    count="size", auto_share="mean"
).round(4).sort_values("auto_share")
print("\nPOSOSTO AUTO ANA KLASI (threshold", round(production_threshold, 4), ")")
print(ana_klasi.to_string())

sensitivity.to_csv("docs/threshold_sensitivity.csv", index=False)
ana_klasi.to_csv("docs/auto_per_class.csv")
