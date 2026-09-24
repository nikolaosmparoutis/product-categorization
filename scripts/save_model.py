"""Train and save the production model, with Platt calibration.

Experiment 04 showed isotonic overfits the small classes, which hold only a
few dozen calibration samples. Platt gives better macro F1, MCC and accuracy,
and cuts the errors that reach the storefront.

What it does, in order:
1. Loads and splits the data with the shared split.
2. Fits the pipeline: brand as a categorical, TF-IDF per text field, LightGBM.
3. Calibrates with Platt on a separate set, without refitting the base model.
4. Measures on test and prints the result row.
5. Saves the joblib and verifies it reloads and predicts.
"""

import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from common import RANDOM_STATE, TFIDF_SETTINGS, TARGET_COLUMN, load_and_split, metrics_row

MODEL_PATH = "model_storage/calibrated_model.joblib"

sets = load_and_split()

model = Pipeline(
    [
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
    ]
)
model.fit(sets["train"], sets["train"][TARGET_COLUMN], lgbm__categorical_feature=[0])

calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
calibrated_model.fit(sets["calibration"], sets["calibration"][TARGET_COLUMN])

probabilities = calibrated_model.predict_proba(sets["test"])
print(metrics_row("final_sigmoid", probabilities, calibrated_model.classes_,
               sets["test"][TARGET_COLUMN].to_numpy(), duration=0))

joblib.dump(calibrated_model, MODEL_PATH)
print("apothikeutike:", MODEL_PATH)

check = joblib.load(MODEL_PATH)
sample_rows = sets["test"].head(3)
print("check fortosis, predictions se raw DataFrame:", check.predict(sample_rows).tolist())
