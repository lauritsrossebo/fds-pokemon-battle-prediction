"""Model definitions and the cross-validation helper used by the notebooks."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def lightgbm_model(**overrides):
    import lightgbm as lgb
    params = dict(n_estimators=800, learning_rate=0.02, num_leaves=31, min_child_samples=80,
                  colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=-1)
    params.update(overrides)
    return lgb.LGBMClassifier(**params)


def xgboost_model(**overrides):
    import xgboost as xgb
    params = dict(n_estimators=700, learning_rate=0.02, max_depth=4, subsample=0.8, colsample_bytree=0.4,
                  min_child_weight=5, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbosity=0)
    params.update(overrides)
    return xgb.XGBClassifier(**params)


def hist_gb_model():
    """HistGradientBoosting configuration used on feature set B."""
    return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, max_leaf_nodes=15, min_samples_leaf=50,
                                          l2_regularization=1.0, max_features=0.5, random_state=SEED)


def logistic_model(C=0.02):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))


def random_forest_model(**overrides):
    params = dict(n_estimators=500, max_features=0.2, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    params.update(overrides)
    return RandomForestClassifier(**params)


class SoftVote(BaseEstimator, ClassifierMixin):
    """Average of the predicted probabilities of several fitted-from-scratch models."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.fitted_ = [clone(m).fit(X, y) for m in self.models]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.fitted_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


def cross_validate_model(model, X, y, n_splits=5, seeds=(SEED,)):
    """Stratified k-fold CV repeated with several seeds.
    Returns the per-fold accuracies and the out-of-fold probabilities, one row per seed."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    scores, oof_all = [], []
    for seed in seeds:
        oof = np.zeros(len(y))
        for trn, val in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
            m = clone(model).fit(X[trn], y[trn])
            oof[val] = m.predict_proba(X[val])[:, 1]
            scores.append(((oof[val] > 0.5) == y[val]).mean())
        oof_all.append(oof)
    return np.array(scores), np.array(oof_all)


def blend_scores(oof_list, y, n_splits=5, seeds=(SEED,)):
    """Per-fold accuracies of the average of several models' out-of-fold probabilities
    (= CV accuracy of the soft-voting ensemble, since all models share the same folds)."""
    y = np.asarray(y)
    p = np.mean(oof_list, axis=0)
    out = []
    for i, seed in enumerate(seeds):
        for _, val in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
            out.append(((p[i][val] > 0.5) == y[val]).mean())
    return np.array(out)
