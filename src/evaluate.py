"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline
on several metrics, out-of-fold probabilities and reports, a fairness check and a paired
comparison of two models. The locked test set is never used here.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def get_metric_scorer(name: str):
    """sklearn scorer by name, plus "fbeta_<beta>" (e.g. fbeta_2: recall weighted 2x precision)."""
    if name.startswith("fbeta_"):
        return make_scorer(fbeta_score, beta=float(name[len("fbeta_"):]), zero_division=0)
    return get_scorer(name)


def metric_scorers(scoring: str, metrics=None) -> dict:
    """{name: scorer} for `scoring` (first) and every other name in `metrics`."""
    names = [scoring] + [m for m in (metrics or []) if m != scoring]
    return {name: get_metric_scorer(name) for name in names}


def cross_validate_pipeline(pipeline, X, y, cv, scoring: str = "accuracy", metrics=None, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, with `scoring` and every metric in `metrics`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the train score, the validation score and their gap
        (for `scoring`), plus `train_<metric>` / `val_<metric>` columns for every metric;
      - oof_proba: out-of-fold probability of the positive class for every row, from the fold
        model that did not train on it (the same fits as the scores). `cv` must be a
        partition (each row validated exactly once).
    """
    scorers = metric_scorers(scoring, metrics)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({
        "fold": range(1, len(scores[f"test_{scoring}"]) + 1),
        "train": scores[f"train_{scoring}"],
        "validation": scores[f"test_{scoring}"],
    })
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[f"train_{name}"] = scores[f"train_{name}"]
        fold_scores[f"val_{name}"] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, scoring: str = "accuracy", title: str = "Cross-validation") -> str:
    """
    Per-fold table (for `scoring`) with mean and std, then every metric's train / validation
    mean, validation std and gap over the folds, as text.
    """
    names = [c[len("val_"):] for c in fold_scores.columns if c.startswith("val_")]
    per_fold = fold_scores.drop(columns=[f"{p}_{n}" for n in names for p in ("train", "val")])
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {scoring})",
        "",
        per_fold.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
    ]
    for col in ["train", "validation", "gap"]:
        sign = "+" if col == "gap" else ""
        lines.append(f"{col.capitalize():<11s} mean = {fold_scores[col].mean():{sign}.3f}   "
                     f"std = {fold_scores[col].std(ddof=1):.3f}")
    if names:
        summary = pd.DataFrame({
            "train": [fold_scores[f"train_{n}"].mean() for n in names],
            "validation": [fold_scores[f"val_{n}"].mean() for n in names],
            "val std": [fold_scores[f"val_{n}"].std(ddof=1) for n in names],
        }, index=names)
        summary["gap"] = summary["train"] - summary["validation"]
        lines += ["", "All metrics (mean over folds):",
                  summary.to_string(float_format=lambda v: f"{v:.3f}")]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """Confusion matrix and classification report of the out-of-fold predictions, as text."""
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str,
                    threshold: float = 0.5) -> str:
    """
    False positive rate (share of true negatives, label 0, predicted positive) per group of
    `sensitive_attr`, for the model's out-of-fold predictions (probability >= `threshold`).
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.
    """
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    predictors = [("Our model", "y_pred_model")]

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of true negatives (label 0) that were predicted positive)",
        "",
    ]
    for label, col in predictors:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)
