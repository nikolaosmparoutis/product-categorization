"""Experiment 06 — can ANY model separate class 5 from class 1049?

The main model makes 171 "truly 1049, predicted 5" errors and 81 the other way.
The hypothesis is that the two categories overlap in the labels. Here a binary
specialist is trained on those two alone, with the full volume of data and a
vocabulary dedicated to one distinction — the best possible conditions.

We measure balanced accuracy and ROC AUC, where chance is 0.50. Plain accuracy
says nothing here: always answering "5" already scores 77%.

If the specialist fails, we have a MEASUREMENT that the distinction is not in
the text, rather than a guess.

What it does, in order:
1. Keeps only the products of classes 5 and 1049.
2. Trains a binary specialist dedicated to that distinction.
3. Measures balanced accuracy and ROC AUC, where chance is 0.50.
4. Compares against the main model on the SAME products and counts what it fixes.
5. Writes a csv.
"""

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from common import RANDOM_STATE, TFIDF_SETTINGS, TARGET_COLUMN, load_and_split

PAIR = [5, 1049]

sets = load_and_split()
training_data = pd.concat([sets["train"], sets["calibration"]])
training_pair = training_data[training_data[TARGET_COLUMN].isin(PAIR)]
test_pair = sets["test"][sets["test"][TARGET_COLUMN].isin(PAIR)]

print("training_data:", len(training_pair), "| test:", len(test_pair))
print(training_pair[TARGET_COLUMN].value_counts().to_string(), "\n")

specialist = Pipeline([
    ("features", ColumnTransformer([
        ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"]),
        ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
        ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
        ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
    ])),
    ("lgbm", LGBMClassifier(
        class_weight="balanced", n_estimators=200, learning_rate=0.05, num_leaves=63,
        min_child_samples=15, subsample=0.75, subsample_freq=1, reg_lambda=0.1, max_bin=60,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )),
])
specialist.fit(training_pair, training_pair[TARGET_COLUMN], lgbm__categorical_feature=[0])

y_true = test_pair[TARGET_COLUMN].to_numpy()
predictions = specialist.predict(test_pair)
probability_1049 = specialist.predict_proba(test_pair)[:, list(specialist.classes_).index(1049)]

majority_baseline = (y_true == 5).mean()
print("MONO O EIDIKOS, sto zeugos 5 kai 1049")
print(f"  accuracy            : {(predictions == y_true).mean():.4f}  (vasi 'panta 5': {majority_baseline:.4f})")
print(f"  balanced accuracy   : {balanced_accuracy_score(y_true, predictions):.4f}  (tyxi: 0,5000)")
print(f"  ROC AUC             : {roc_auc_score(y_true == 1049, probability_1049):.4f}  (tyxi: 0,5000)")
print()
print(classification_report(y_true, predictions, digits=3))

main_model = joblib.load("model_storage/calibrated_model.joblib")
main_predictions = main_model.classes_[main_model.predict_proba(test_pair).argmax(axis=1)]
main_model_wrong = (main_predictions != y_true) & pd.Series(main_predictions, index=test_pair.index).isin(PAIR).to_numpy()

print("SYGKRISI ME TO KYRIO MONTELO, sta idia proionta")
print(f"  lathi kyriou montelou        : {int(main_model_wrong.sum())}")
print(f"  apo auta, o specialist diorthonei: {int((predictions[main_model_wrong] == y_true[main_model_wrong]).sum())}")
print(f"  lathi tou eidikou synolika   : {int((predictions != y_true).sum())}")

pd.DataFrame([{
    "experiment": "specialist_5_vs_1049",
    "accuracy": round((predictions == y_true).mean(), 4),
    "baseline_always_5": round(majority_baseline, 4),
    "balanced_accuracy": round(balanced_accuracy_score(y_true, predictions), 4),
    "roc_auc": round(roc_auc_score(y_true == 1049, probability_1049), 4),
    "specialist_errors": int((predictions != y_true).sum()),
    "main_model_errors": int(main_model_wrong.sum()),
    "fixed_by_specialist": int((predictions[main_model_wrong] == y_true[main_model_wrong]).sum()),
}]).to_csv("docs/experiment_06_specialist.csv", index=False)
