"""Shared setup for every experiment: same dedup, same split, same metrics.

What it does, in order:
1. Loads the csv and drops exact duplicate rows.
2. Splits into train, validation, calibration, test — stratified AND group-aware.
3. Computes the "balanced" class weights as a dict, so they can be altered by hand.
4. Exposes one metrics function, so every experiment is measured identically.
"""

import time

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

TARGET_COLUMN = "category"
TEXT_COLUMNS = ["product_name", "product_description", "features"]
RANDOM_STATE = 42
COST_REVIEW = 20 / 3600 * 10
COST_ERROR = 1.0
MAJORITY_CLASS = 5
DATA_PATH = "notebooks/data/raw/products.csv"

TFIDF_SETTINGS = dict(
    ngram_range=(1, 2), min_df=3, max_features=10_000, sublinear_tf=True, stop_words="english"
)


def load_and_split() -> dict[str, pd.DataFrame]:
    """Load the csv, drop exact duplicates, group-split into train/calibration/test.

    Group = (product_description, features), so that variants of the same text
    never straddle train and test.
    """
    data = pd.read_csv(DATA_PATH).drop_duplicates()
    data[TEXT_COLUMNS] = data[TEXT_COLUMNS].fillna("")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    text_groups = data.groupby(["product_description", "features"]).ngroup()
    theseis_ekpaideusis, idx_test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE).split(
            data, groups=text_groups
        )
    )
    data_fitting = data.iloc[theseis_ekpaideusis]
    theseis_train, idx_calibration = next(
        GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE).split(
            data_fitting, groups=text_groups.iloc[theseis_ekpaideusis]
        )
    )
    return {
        "train": data_fitting.iloc[theseis_train],
        "calibration": data_fitting.iloc[idx_calibration],
        "test": data.iloc[idx_test],
    }


def load_and_split_4(
    pososto_test: float = 0.20, pososto_calibration: float = 0.075, pososto_validation: float = 0.075
) -> dict[str, pd.DataFrame]:
    """Four-way split, stratified AND group-aware at every level.

    Shares are of the whole dataset, not nested: the defaults give
    65 train / 20 test / 7.5 validation / 7.5 calibration.

    `StratifiedGroupKFold` keeps text groups intact — otherwise test would
    measure text the model has already seen — while balancing the classes, so
    the small ones do not end up with a handful of samples in validation or
    calibration.

    Validation exists so that decisions (hyperparameters, calibration method,
    ablations) are never made by looking at test. Test is opened once, at the end.
    """
    data = pd.read_csv(DATA_PATH).drop_duplicates()
    data[TEXT_COLUMNS] = data[TEXT_COLUMNS].fillna("")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    groups_col = data.groupby(["product_description", "features"]).ngroup()

    def cut_fold(pleuro: pd.DataFrame, omades_pleuras: pd.Series, pososto: float):
        """Cut one fold of roughly `share`, keeping groups intact and classes balanced."""
        n_folds = max(2, round(1 / pososto))
        splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        theseis_a, idx_b = next(splitter.split(pleuro, pleuro[TARGET_COLUMN], groups=omades_pleuras))
        return pleuro.iloc[theseis_a], pleuro.iloc[idx_b], omades_pleuras.iloc[theseis_a]

    ypoloipo, test, remaining_groups = cut_fold(data, groups_col, pososto_test)
    meta_to_test = 1 - pososto_test
    ypoloipo, calibration, remaining_groups = cut_fold(
        ypoloipo, remaining_groups, pososto_calibration / meta_to_test
    )
    train, validation, _ = cut_fold(
        ypoloipo, remaining_groups, pososto_validation / (meta_to_test - pososto_calibration)
    )

    return {"train": train, "validation": validation, "calibration": calibration, "test": test}


def balanced_weights(data_train: pd.DataFrame) -> dict[int, float]:
    """The weights class_weight="balanced" would compute, as a dict we can alter by hand."""
    count_per_class = data_train[TARGET_COLUMN].value_counts()
    return {
        cls: len(data_train) / (len(count_per_class) * count)
        for cls, count in count_per_class.items()
    }


def metrics_row(label: str, probabilities, classes_list, y_true, duration: float) -> dict:
    """One row of results, derived from the test probabilities.

    Covers macro F1/precision, MCC, accuracy, the automatic share under the
    cost-based threshold, and the count of errors where the true class is 1049
    and the prediction is 5.
    """
    predictions = classes_list[probabilities.argmax(axis=1)]

    is_auto = (1 - probabilities.max(axis=1)) * COST_ERROR <= COST_REVIEW
    correct_auto = predictions[is_auto] == y_true[is_auto]

    return {
        "experiment": label,
        "macro_f1": round(f1_score(y_true, predictions, average="macro"), 4),
        "macro_precision": round(precision_score(y_true, predictions, average="macro", zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, predictions), 4),
        "accuracy": round((predictions == y_true).mean(), 4),
        "auto_share": round(is_auto.mean(), 4),
        "precision_on_auto": round(correct_auto.mean(), 5),
        "storefront_errors": int((~correct_auto).sum()),
        "errors_1049_as_5": int(((y_true == 1049) & (predictions == 5)).sum()),
        "errors_5_as_1049": int(((y_true == 5) & (predictions == 1049)).sum()),
        "errors_into_5_total": int(((y_true != 5) & (predictions == 5)).sum()),
        "recall_class_5": round(
            ((predictions == 5) & (y_true == 5)).sum() / (y_true == 5).sum(), 4
        ),
        "duration_sec": round(duration),
    }


def train_and_evaluate(label: str, model, sets: dict[str, pd.DataFrame], **fit_params) -> dict:
    """Fit, calibrate on a separate set, and return metrics on test."""
    started = time.time()
    model.fit(sets["train"], sets["train"][TARGET_COLUMN], **fit_params)
    duration = time.time() - started

    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated.fit(sets["calibration"], sets["calibration"][TARGET_COLUMN])
    return metrics_row(
        label,
        calibrated.predict_proba(sets["test"]),
        calibrated.classes_,
        sets["test"][TARGET_COLUMN].to_numpy(),
        duration,
    )
