"""Experiment 07 — hyperparameter tuning with Optuna on a validation set.

Until now the hyperparameters came from experience, with no search, and every
comparison was made on test. A fourth set is introduced here: validation takes
every decision and test stays sealed.

Speed: TF-IDF does not change between trials, so the ColumnTransformer is fit
ONCE on train and the trials work on the ready sparse matrices.

Early stopping runs on the evaluation set, so n_estimators picks itself instead
of entering the search.

What it does, in order:
1. Splits into four sets. Validation AND calibration are merged for measurement
   only: with 7,819 products alone, class 9 collapsed to F1 0.384 because the
   group constraint handed it one unusual text group, and Optuna would have
   tuned to that. Merged they give 15,136 products and twice the samples in the
   small classes, at no extra cost in time.
2. Fits TF-IDF ONCE, so no trial recomputes it.
3. Runs Optuna on macro F1, with early stopping choosing n_estimators.
4. On a tie inside the noise, picks the SIMPLEST model, not the lucky one.
5. Writes the best parameters and every trial.
"""

import json
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder

from common import RANDOM_STATE, TFIDF_SETTINGS, TARGET_COLUMN, load_and_split_4

N_TRIALS = 20
MAX_TREES = 1500
PATIENCE = 30

sets = load_and_split_4()

evaluation = pd.concat([sets["validation"], sets["calibration"]])
targets = {"train": sets["train"][TARGET_COLUMN].to_numpy(), "evaluation": evaluation[TARGET_COLUMN].to_numpy()}

transformer = ColumnTransformer([
    ("brand", OrdinalEncoder(min_frequency=20, handle_unknown="use_encoded_value", unknown_value=-1), ["brand"]),
    ("name", TfidfVectorizer(**TFIDF_SETTINGS), "product_name"),
    ("description", TfidfVectorizer(**TFIDF_SETTINGS), "product_description"),
    ("features", TfidfVectorizer(**TFIDF_SETTINGS), "features"),
])
started = time.time()
X_train = transformer.fit_transform(sets["train"])
X_aksiologisis = transformer.transform(evaluation)
print(f"features etoima se {time.time() - started:.0f}s | train {X_train.shape} | evaluation {X_aksiologisis.shape}", flush=True)


def macro_f1_eval(y_true, probabilities):
    """Custom eval metric for early stopping.

    On logloss, training stopped at 19 to 101 trees: the weighted logloss
    plateaus early while macro F1 is still climbing.

    Supplying this function is not enough on its own. LightGBM also computes
    the default multi_logloss, and early stopping fires as soon as ANY metric
    stops improving. It needs metric="None" on the model and
    first_metric_only=True on the callback.
    """
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(len(y_true), -1, order="F")
    predictions = np.unique(y_true)[probabilities.argmax(axis=1)]
    return "macro_f1", f1_score(y_true, predictions, average="macro"), True


def objective(trial: optuna.Trial) -> float:
    """Macro F1 on validation plus calibration. Test is never touched."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 32, 255, log=True),
    }
    model = lgb.LGBMClassifier(
        class_weight="balanced", n_estimators=MAX_TREES, subsample_freq=1,
        metric="None", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, **params,
    )
    model.fit(
        X_train, targets["train"],
        eval_set=[(X_aksiologisis, targets["evaluation"])], eval_metric=macro_f1_eval,
        categorical_feature=[0],
        callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False)],
    )
    trial.set_user_attr("trees", model.best_iteration_)
    return f1_score(targets["evaluation"], model.predict(X_aksiologisis), average="macro")


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

def report_trial(study_, trial_):
    print(f"trial {trial_.number:3d} | macro F1 {trial_.value:.4f} | best {study_.best_value:.4f} "
          f"| trees {trial_.user_attrs.get('trees')}", flush=True)

study.optimize(objective, n_trials=N_TRIALS, callbacks=[report_trial])

NOISE = 0.004

def complexity(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
    """How free a model is: many leaves per sample and little regularisation."""
    return trial.params["num_leaves"] / trial.params["min_child_samples"], -trial.params["reg_lambda"]


finished = [(t, float(t.value)) for t in study.trials if t.value is not None]
candidates = [(t, v) for t, v in finished if study.best_value - v <= NOISE]
epilogi, chosen_value = min(candidates, key=lambda zeugos: complexity(zeugos[0]))

print(f"\nTrials mesa ston thoryvo ({NOISE}) tou kalyterou: {len(candidates)}")
for t, v in sorted(candidates, key=lambda zeugos: -zeugos[1])[:5]:
    marker = " <-- epilegmeno" if t.number == epilogi.number else ""
    print(f"  trial {t.number:3d} | macro F1 {v:.4f} | leaves {t.params['num_leaves']:3d} "
          f"| min_child {t.params['min_child_samples']:3d} | reg_lambda {t.params['reg_lambda']:.3f}{marker}")

best_params = dict(epilogi.params)
best_params["n_estimators"] = epilogi.user_attrs["trees"]
print(f"\nEPILEGMENA PARAMS (validation macro F1 {chosen_value:.4f}, best itan {study.best_value:.4f}):")
print(json.dumps(best_params, indent=2))

with open("docs/optuna_best_params.json", "w") as f:
    json.dump({
        "validation_macro_f1": round(chosen_value, 4),
        "best_trial_macro_f1": round(study.best_value, 4),
        "trials_within_noise": len(candidates),
        "params": best_params,
    }, f, indent=2)
study.trials_dataframe().to_csv("docs/experiment_07_optuna_trials.csv", index=False)
