"""The three final submissions.

Each recipe refits its models on all 10,000 training battles and writes one CSV.

    python -m pokefds.submissions blend_lr_hgb --out some_folder

The recipes are kept exactly as they were when the files were uploaded, including small
differences between them (for example how the feature table is passed to the models).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features_b
from .data import find_data_dir, load_jsonl
from .features import TypeChart, build_species_dictionary
from .features_ext import build_vocab, create_features_ext
from .models import SEED, hist_gb_model, lightgbm_model, logistic_model, xgboost_model

ID_COLS = ("battle_id", "player_won")


def load_competition_data(data_dir=None):
    data_dir = Path(data_dir) if data_dir else find_data_dir()
    return {"dir": data_dir,
            "train": load_jsonl(data_dir / "train.jsonl"),
            "test": load_jsonl(data_dir / "test.jsonl"),
            "sample": pd.read_csv(data_dir / "sample_submission.csv"),
            "type_chart": pd.read_csv(data_dir / "type_chart_gen2.csv")}


def feature_set_a(data):
    """Feature set A (291 columns): team statistics, 30-turn dynamics, turn-30 snapshot, bag of moves."""
    chart = TypeChart(data["dir"] / "type_chart_gen2.csv")
    species = build_species_dictionary(data["train"])      # train only
    vocab = build_vocab(data["train"])                      # train only
    A_tr = create_features_ext(data["train"], species, chart, vocab)
    A_te = create_features_ext(data["test"], species, chart, vocab)
    A_cols = [c for c in A_tr.columns if c not in ID_COLS and not c.startswith("sp_")]
    return A_tr, A_te, A_cols


def feature_set_b(data):
    """Feature set B (233 columns): last known state of every Pokémon after the 30 turns."""
    features_b.init(data["train"], data["type_chart"])      # lookup tables from train only
    B_tr = features_b.create_features(data["train"])
    B_te = features_b.create_features(data["test"])
    B_cols = [c for c in B_tr.columns if c not in ID_COLS]
    return B_tr, B_te, B_cols


def as_arrays(tr, te, cols):
    X = tr[cols].astype(float).values
    X_te = te.reindex(columns=cols, fill_value=0).astype(float).values
    return X, X_te


def save_submission(proba, battle_ids, sample, path):
    """Threshold the probabilities at 0.5 and write them in the order of sample_submission.csv."""
    pred = pd.Series((np.asarray(proba) > 0.5).astype(int), index=battle_ids)
    submission = sample[["battle_id"]].copy()
    submission["player_won"] = submission["battle_id"].map(pred).astype(int)
    submission.to_csv(path, index=False)
    print(f"{Path(path).name}: {len(submission)} rows, share of P1 wins {submission['player_won'].mean():.3f}")
    return submission


def blend_lr_hgb(data, out_dir):
    """Submission 1: logistic regression + HistGradientBoosting on feature set B."""
    B_tr, B_te, B_cols = feature_set_b(data)
    y = B_tr["player_won"].astype(int).values
    XB, XB_te = as_arrays(B_tr, B_te, B_cols)
    proba = [make().fit(XB, y).predict_proba(XB_te)[:, 1] for make in (logistic_model, hist_gb_model)]
    return save_submission(np.mean(proba, axis=0), B_te["battle_id"], data["sample"],
                           Path(out_dir) / "submission_blend_lr_hgb.csv")


def blend_first_version(data, out_dir):
    """Submission 2: the first version of the same blend, as it was run in the original notebook
    (models fitted on the feature DataFrame, logistic regression with max_iter=5000)."""
    B_tr, B_te, B_cols = feature_set_b(data)
    B_te = B_te.reindex(columns=["battle_id"] + B_cols, fill_value=0)
    X, y, X_te = B_tr[B_cols], B_tr["player_won"], B_te[B_cols]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=5000)).fit(X, y)
    hgb = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, max_leaf_nodes=15,
                                         min_samples_leaf=50, l2_regularization=1.0,
                                         max_features=0.5, random_state=SEED).fit(X, y)
    proba = (lr.predict_proba(X_te)[:, 1] + hgb.predict_proba(X_te)[:, 1]) / 2
    return save_submission(proba, B_te["battle_id"], data["sample"], Path(out_dir) / "submission_blend.csv")


def merged_ensemble(data, out_dir):
    """Submission 3: LR + HGB on feature set B and LightGBM + XGBoost on feature set A, averaged."""
    A_tr, A_te, A_cols = feature_set_a(data)
    B_tr, B_te, B_cols = feature_set_b(data)
    assert (A_tr["battle_id"].values == B_tr["battle_id"].values).all()
    y = A_tr["player_won"].astype(int).values
    XA, XA_te = as_arrays(A_tr, A_te, A_cols)
    XB, XB_te = as_arrays(B_tr, B_te, B_cols)
    runs = [(logistic_model, XB, XB_te), (hist_gb_model, XB, XB_te),
            (lightgbm_model, XA, XA_te), (xgboost_model, XA, XA_te)]
    proba = [make().fit(X, y).predict_proba(X_te)[:, 1] for make, X, X_te in runs]
    return save_submission(np.mean(proba, axis=0), A_te["battle_id"], data["sample"],
                           Path(out_dir) / "submission_merged_ensemble.csv")


SUBMISSIONS = {
    "blend_lr_hgb": (blend_lr_hgb, "submission_blend_lr_hgb.csv"),
    "blend_first_version": (blend_first_version, "submission_blend.csv"),
    "merged_ensemble": (merged_ensemble, "submission_merged_ensemble.csv"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=list(SUBMISSIONS))
    ap.add_argument("--out", default=".", help="folder where the CSV is written")
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    SUBMISSIONS[args.name][0](load_competition_data(args.data_dir), args.out)


if __name__ == "__main__":
    main()
