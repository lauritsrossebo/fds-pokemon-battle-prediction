"""Evaluation helpers: metric tables and the figures shown in the notebook and in the report.

All metrics are computed on out-of-fold predictions from the cross-validation, so every
training battle is scored by a model that did not see it.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold

from .models import SEED, hist_gb_model, logistic_model

# Colours: one fixed colour per series, grey for reference lines, dark grey for text.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = [BLUE, ORANGE, AQUA]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "text.color": INK, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _finish(fig, save_path):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# --------------------------------------------------------------------------- #
# Simple reference models
# --------------------------------------------------------------------------- #
def starter_features(battles):
    """The six features of the course starter notebook: mean base stats of P1's team and P2's lead."""
    rows = []
    for b in battles:
        team, lead = b["p1_team_details"], b["p2_lead_details"]
        rows.append({"p1_mean_hp": np.mean([p["base_hp"] for p in team]),
                     "p1_mean_spe": np.mean([p["base_spe"] for p in team]),
                     "p1_mean_atk": np.mean([p["base_atk"] for p in team]),
                     "p1_mean_def": np.mean([p["base_def"] for p in team]),
                     "p2_lead_hp": lead["base_hp"], "p2_lead_spe": lead["base_spe"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def fold_accuracies(proba, y, seed=SEED, n_splits=5):
    """Accuracy of out-of-fold probabilities on each validation fold."""
    y = np.asarray(y)
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)
    return np.array([((proba[val] > 0.5) == y[val]).mean() for _, val in folds])


def metrics_table(probas, y, seed=SEED):
    """One row per model: accuracy (mean and std over the folds) and other classification metrics."""
    y = np.asarray(y)
    rows = {}
    for name, p in probas.items():
        pred = (p > 0.5).astype(int)
        acc = fold_accuracies(p, y, seed)
        rows[name] = {"accuracy": accuracy_score(y, pred), "fold std": acc.std(),
                      "ROC AUC": roc_auc_score(y, p), "log loss": log_loss(y, p),
                      "Brier": brier_score_loss(y, p), "precision": precision_score(y, pred),
                      "recall": recall_score(y, pred), "F1": f1_score(y, pred)}
    return pd.DataFrame(rows).T


def confusion_table(proba, y):
    cm = confusion_matrix(np.asarray(y), (proba > 0.5).astype(int))
    return pd.DataFrame(cm, index=["actual: P1 lost", "actual: P1 won"],
                        columns=["predicted: P1 lost", "predicted: P1 won"])


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def plot_roc(probas, y, title="ROC curves (out-of-fold predictions)", save_path=None):
    """ROC curve of up to three models. The legend gives the area under each curve."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="random guess (AUC 0.50)")
    for (name, p), color in zip(probas.items(), SERIES):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title,
           xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
    ax.legend(loc="lower right", fontsize=9)
    return _finish(fig, save_path)


def plot_calibration(proba, y, n_bins=10, title="Calibration of the predicted win probability", save_path=None):
    """Top: share of P1 wins among battles with a similar predicted probability.
    Bottom: how many battles fall in each probability bin."""
    y = np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(proba, edges) - 1, 0, n_bins - 1)
    centers, observed, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(m.sum())
        if m.sum() >= 20:
            centers.append(proba[m].mean())
            observed.append(y[m].mean())
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.6, 5.4), sharex=True, height_ratios=[3, 1])
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--", label="perfect calibration")
    ax.plot(centers, observed, color=BLUE, marker="o", label="model")
    ax.set(ylabel="Share of battles P1 won", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=9)
    ax2.bar((edges[:-1] + edges[1:]) / 2, np.array(counts) / len(y), width=0.09, color=BLUE)
    ax2.set(xlabel="Predicted probability that P1 wins", ylabel="Share of battles")
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    return _finish(fig, save_path)


def plot_accuracy_by_confidence(proba, y, title="Accuracy by confidence of the prediction", save_path=None):
    """Accuracy of the battles grouped by how far the predicted probability is from 0.5."""
    y = np.asarray(y)
    conf = np.maximum(proba, 1 - proba)
    correct = (proba > 0.5) == y
    edges = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0001]
    labels, acc, share = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        labels.append(f"{lo:.0%}–{min(hi, 1):.0%}")
        acc.append(correct[m].mean() if m.any() else np.nan)
        share.append(m.mean())
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(labels, acc, width=0.7, color=BLUE, zorder=3)
    for bar, a, s in zip(bars, acc, share):
        ax.text(bar.get_x() + bar.get_width() / 2, a - 0.06, f"{a:.0%}", ha="center", fontsize=10,
                color="white", fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"{s:.0%} of\nbattles", ha="center",
                fontsize=8, color="white")
    ax.axhline(correct.mean(), color=INK_2, linewidth=1, linestyle="--", zorder=2)
    ax.text(-0.45, correct.mean() + 0.012, f"overall accuracy {correct.mean():.1%}", ha="left",
            fontsize=9, color=INK_2)
    ax.set(xlabel="Predicted probability of the predicted winner", ylabel="Accuracy", title=title, ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x", visible=False)
    return _finish(fig, save_path)


def learning_curve_blend(X, y, sizes=(500, 1000, 2000, 4000, 8000), seed=SEED, n_splits=5):
    """Accuracy of the LR + HGB blend when it is trained on only part of each training fold.
    Returns a table with the mean and std of the training and validation accuracy per size."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    rng = np.random.default_rng(seed)
    rows = []
    for trn, val in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        trn = rng.permutation(trn)
        for n in sizes:
            sub = trn[:n]
            models = [make().fit(X[sub], y[sub]) for make in (logistic_model, hist_gb_model)]
            p_tr = np.mean([m.predict_proba(X[sub])[:, 1] for m in models], axis=0)
            p_val = np.mean([m.predict_proba(X[val])[:, 1] for m in models], axis=0)
            rows.append({"n_train": len(sub), "train": ((p_tr > 0.5) == y[sub]).mean(),
                         "validation": ((p_val > 0.5) == y[val]).mean()})
    out = pd.DataFrame(rows).groupby("n_train").agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def plot_learning_curve(curve, title="Learning curve of the LR + HGB blend", save_path=None):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for col, color, label in (("train", ORANGE, "training battles"), ("validation", BLUE, "validation fold")):
        mean, std = curve[f"{col}_mean"], curve[f"{col}_std"]
        ax.fill_between(curve["n_train"], mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(curve["n_train"], mean, color=color, marker="o", label=label)
        ax.annotate(f"{mean.iloc[-1]:.1%}", (curve["n_train"].iloc[-1], mean.iloc[-1]),
                    textcoords="offset points", xytext=(8, -3), fontsize=9)
    ax.set(xlabel="Number of training battles", ylabel="Accuracy", title=title,
           xlim=(0, curve["n_train"].max() * 1.12))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper right", fontsize=9)
    return _finish(fig, save_path)


def plot_scores_by_stage(stages, title="Cross-validation and leaderboard accuracy of each submission",
                         save_path=None):
    """Dot plot with one row per submission and one dot per score (CV, public, private)."""
    fig, ax = plt.subplots(figsize=(6.6, 0.55 * len(stages) + 1.8))
    ypos = np.arange(len(stages))[::-1]
    for col, color, marker in (("CV", BLUE, "o"), ("public", ORANGE, "s"), ("private", AQUA, "D")):
        ax.scatter(stages[col], ypos, color=color, marker=marker, s=55, zorder=3, label=col,
                   edgecolor=SURFACE, linewidth=1.2)
    ax.set_yticks(ypos, stages["submission"])
    ax.set(xlabel="Accuracy", title=title, ylim=(-1.3, len(stages) - 0.4))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", ncols=3, fontsize=9, handletextpad=0.2, columnspacing=1.0)
    return _finish(fig, save_path)


def plot_top_coefficients(coefs, k=15, title="Largest logistic regression coefficients (feature set B)",
                          save_path=None):
    """Horizontal bars of the k largest standardised coefficients. Blue pushes towards a P1 win, red against."""
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index)[:k][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 0.3 * k + 1.2))
    ax.barh(top.index, top.values, height=0.7, color=[BLUE if v > 0 else RED for v in top.values])
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set(xlabel="Standardised coefficient (positive: P1 more likely to win)", title=title)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=9)
    return _finish(fig, save_path)
