"""Model pipelines, nested cross-validation and metrics.

Every step that learns from data (imputation, scaling, hyperparameter choice,
early stopping) happens inside the training part of a fold, so the held-out
fold never influences the model that is scored on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

LABEL = "In-hospital_death"
NON_FEATURES = ["RecordID", "set", "split", "fold", LABEL, "SAPS-I", "SOFA"]
RANDOM_STATE = 42

# Right-skewed, positive variables (from 03_eda). Their value features and all
# count features get log1p before scaling in logistic regression.
LOG_VARS = ["ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose",
            "Lactate", "Platelets", "TroponinI", "TroponinT", "Urine", "WBC"]


# ── data ────────────────────────────────────────────────────────────────
def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def empty_stay_mask(df: pd.DataFrame) -> pd.Series:
    """True for stays with no time-series measurement at all (all *_count == 0)."""
    counts = [c for c in df.columns if c.endswith("_count")]
    return df[counts].sum(axis=1) == 0


# ── pipelines ───────────────────────────────────────────────────────────
def log_columns(features: list[str]) -> list[str]:
    value_stats = ("_min", "_max", "_mean", "_last")
    return [f for f in features
            if f.endswith("_count")
            or (f.endswith(value_stats) and f.rsplit("_", 1)[0] in LOG_VARS)]


def _log1p_nonneg(x):
    return np.log1p(np.clip(x, 0, None))


def make_logreg(features: list[str], class_weight=None, C: float = 0.1) -> Pipeline:
    """log1p skewed columns -> median impute -> standardise -> L2 logistic regression.

    The has_<var> flags are already features, so imputing a median does not hide
    that a value was missing.
    """
    logs = log_columns(features)
    rest = [f for f in features if f not in logs]
    pre = ColumnTransformer(
        [("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one"), logs),
         ("keep", "passthrough", rest)],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("log", pre),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)),
    ])


def make_rf(**params) -> RandomForestClassifier:
    """Random forest. scikit-learn >= 1.4 trees handle NaN natively, so no imputation."""
    defaults = dict(n_estimators=400, class_weight="balanced_subsample",
                    n_jobs=-1, random_state=RANDOM_STATE)
    return RandomForestClassifier(**{**defaults, **params})


class LGBMEarlyStopping(BaseEstimator, ClassifierMixin):
    """LightGBM with early stopping on a slice of the TRAINING data.

    fit() holds out `val_size` of whatever data it receives (stratified), trains
    up to `n_estimators` rounds, and keeps the round with the best validation
    log loss. The caller's held-out fold is never seen.
    """

    def __init__(self, num_leaves=15, min_child_samples=50, learning_rate=0.03,
                 colsample_bytree=0.5, subsample=0.8, reg_lambda=1.0,
                 n_estimators=3000, val_size=0.15, patience=100, random_state=RANDOM_STATE):
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.learning_rate = learning_rate
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.n_estimators = n_estimators
        self.val_size = val_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        import lightgbm as lgb
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_size, stratify=y, random_state=self.random_state)
        self.model_ = lgb.LGBMClassifier(
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            learning_rate=self.learning_rate, colsample_bytree=self.colsample_bytree,
            subsample=self.subsample, subsample_freq=1, reg_lambda=self.reg_lambda,
            n_estimators=self.n_estimators, random_state=self.random_state,
            verbose=-1, n_jobs=-1)
        import warnings
        with warnings.catch_warnings():
            # lightgbm 4.7 renamed eval_set; the old name still works and keeps >=4.3 compatible
            warnings.simplefilter("ignore")
            self.model_.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="binary_logloss",
                            callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        self.best_iteration_ = self.model_.best_iteration_
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X, num_iteration=self.best_iteration_)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_


# ── cross-validation ────────────────────────────────────────────────────
def nested_cv(estimator, grid: dict | None, X: pd.DataFrame, y: pd.Series, folds: pd.Series,
              inner_splits: int = 3, scoring: str = "average_precision"):
    """Outer loop = the saved folds. Inner loop = grid search on the outer training part.

    Returns (out-of-fold predicted probabilities, per-fold table).
    """
    oof = pd.Series(np.nan, index=X.index)
    rows = []
    for k in sorted(folds.unique()):
        tr, te = folds != k, folds == k
        if grid:
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=RANDOM_STATE)
            search = GridSearchCV(clone(estimator), grid, scoring=scoring, cv=inner, n_jobs=1)
            search.fit(X[tr], y[tr])
            model, params = search.best_estimator_, search.best_params_
        else:
            model, params = clone(estimator).fit(X[tr], y[tr]), {}
        p = model.predict_proba(X[te])[:, 1]
        oof[te] = p
        rows.append({"fold": k, "n": int(te.sum()), "AUROC": roc_auc_score(y[te], p),
                     "AUPRC": average_precision_score(y[te], p), "params": params})
    return oof, pd.DataFrame(rows)


# ── metrics ─────────────────────────────────────────────────────────────
def bootstrap_ci(y, p, metric, n: int = 1000, seed: int = RANDOM_STATE):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(metric(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])


def summarise(y, p, n_boot: int = 1000) -> dict:
    """AUROC, AUPRC (with 95% bootstrap CIs) and Brier score for one set of predictions."""
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": len(y), "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p)}
    out["AUROC_lo"], out["AUROC_hi"] = bootstrap_ci(y, p, roc_auc_score, n_boot)
    out["AUPRC_lo"], out["AUPRC_hi"] = bootstrap_ci(y, p, average_precision_score, n_boot)
    out["Brier"] = brier_score_loss(y, p) if p.min() >= 0 and p.max() <= 1 else np.nan
    return out


def paired_bootstrap_diff(y, p_a, p_b, metric=roc_auc_score, n: int = 2000, seed: int = RANDOM_STATE):
    """Metric(a) - metric(b) on the same resampled patients each time.

    Returns (observed difference, 2.5th, 97.5th percentile, share of resamples where a <= b).
    """
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(metric(y[i], p_a[i]) - metric(y[i], p_b[i]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return metric(y, p_a) - metric(y, p_b), lo, hi, float((diffs <= 0).mean())
