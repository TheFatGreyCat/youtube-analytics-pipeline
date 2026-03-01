"""
Training module for YouTube viral video prediction.

Splits data with temporal awareness, tunes an XGBoost classifier with
Optuna, trains the final model, and evaluates it on a held-out test set.
1
Pipeline order:
    data_loader.py  →  label.py  →  features.py  →  train.py  →  save_load.py

Usage (standalone):
    python -m ml.train
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PLOTS_DIR = Path(__file__).parent.parent / "plots"


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 1 — TIME-BASED SPLIT
# ═════════════════════════════════════════════════════════════════════════════

def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    published_at: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Chronological train / val / test split — no random shuffle.

    Sorting by ``published_at`` prevents temporal data leakage where a
    future video's engagement metrics could influence training on earlier
    videos.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target (is_viral).
    published_at : pd.Series
        Publish timestamp for each video — used only for sorting.
    train_ratio : float, optional
        Fraction of data used for training. Default 0.70.
    val_ratio : float, optional
        Fraction of data used for validation. Default 0.15.
        test_ratio = 1 - train_ratio - val_ratio.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test

    Raises
    ------
    ValueError
        If X / y are empty or positive samples in train < 50.
    """
    if X.empty or y.empty:
        raise ValueError("X or y is empty — cannot split.")
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have the same length.")

    n_pos = int(y.sum())
    if n_pos < 10:
        raise ValueError(
            f"Only {n_pos} positive (viral) samples found. "
            "Need at least 10. Review label definition in ml/label.py."
        )

    # Sort by publish time
    sort_idx = (
        pd.to_datetime(published_at, utc=True, errors="coerce")
        .fillna(pd.Timestamp.min.tz_localize("UTC"))
        .argsort()
        .values
    )
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)
    ts_sorted = (
        pd.to_datetime(published_at, utc=True, errors="coerce")
        .iloc[sort_idx]
        .reset_index(drop=True)
    )

    n = len(X_sorted)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    X_train, y_train = X_sorted.iloc[:train_end],      y_sorted.iloc[:train_end]
    X_val,   y_val   = X_sorted.iloc[train_end:val_end], y_sorted.iloc[train_end:val_end]
    X_test,  y_test  = X_sorted.iloc[val_end:],         y_sorted.iloc[val_end:]
    ts_train = ts_sorted.iloc[:train_end]
    ts_val   = ts_sorted.iloc[train_end:val_end]
    ts_test  = ts_sorted.iloc[val_end:]

    def _fmt(ts: pd.Series) -> tuple[str, str]:
        valid = ts.dropna()
        if valid.empty:
            return "N/A", "N/A"
        return str(valid.min().date()), str(valid.max().date())

    def _vrate(s: pd.Series) -> float:
        return float(s.mean() * 100) if len(s) > 0 else 0.0

    print()
    print("=" * 60)
    print(" TIME-BASED SPLIT")
    print("=" * 60)
    for label, xi, yi, ts in [
        ("Train", X_train, y_train, ts_train),
        ("Val  ", X_val,   y_val,   ts_val),
        ("Test ", X_test,  y_test,  ts_test),
    ]:
        mn, mx = _fmt(ts)
        print(f"  {label}: {len(xi):,} videos  ({len(xi)/n*100:.0f}%)")
        print(f"         Tu {mn} den {mx}")
        print(f"         Viral rate: {_vrate(yi):.1f}%")
    print("=" * 60)

    # Warnings
    train_vr = _vrate(y_train)
    test_vr  = _vrate(y_test)
    delta    = abs(train_vr - test_vr)
    if delta > 5:
        logger.warning(
            "[CANH BAO] Viral rate lech %.1f%% giua train (%.1f%%) va test (%.1f%%). "
            "Co the do seasonal effect. Kiem tra phan bo theo thoi gian.",
            delta, train_vr, test_vr,
        )
    n_pos_train = int(y_train.sum())
    if n_pos_train < 50:
        logger.warning(
            "[CANH BAO] Chi co %d video viral trong train. "
            "Model co the khong hoc duoc du pattern.",
            n_pos_train,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — OPTUNA HYPERPARAMETER TUNING
# ═════════════════════════════════════════════════════════════════════════════

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Search for the best XGBoost hyperparameters using Optuna TPE.

    Maximises ROC-AUC on the validation set.  ``scale_pos_weight`` is fixed
    before the search and passed through to every trial.

    Parameters
    ----------
    X_train, y_train : training split.
    X_val, y_val     : validation split used to compute the objective.
    n_trials         : number of Optuna trials. Default 50.
    random_state     : reproducibility seed. Default 42.

    Returns
    -------
    dict
        Best hyperparameter dict (includes scale_pos_weight).
    """
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("No positive samples in y_train — cannot compute scale_pos_weight.")
    scale_pos_weight = n_neg / n_pos
    logger.info(
        "scale_pos_weight = %.2f  (%d non-viral / %d viral)",
        scale_pos_weight, n_neg, n_pos,
    )

    no_improve_count = [0]
    best_val_so_far  = [0.0]

    def objective(trial: optuna.Trial) -> float:
        params: dict = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1500),
            "max_depth":          trial.suggest_int("max_depth", 3, 9),
            "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 20),
            "gamma":              trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":   scale_pos_weight,
            "eval_metric":        "auc",
            "early_stopping_rounds": 50,
            "random_state":       random_state,
            "tree_method":        "hist",
        }
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, preds)

        # Track no-improve warning
        if score > best_val_so_far[0]:
            best_val_so_far[0] = score
            no_improve_count[0] = 0
        else:
            no_improve_count[0] += 1
            if no_improve_count[0] == 20:
                logger.warning(
                    "[CANH BAO] Optuna khong cai thien sau 20 trials lien tiep. "
                    "Hien tai best ROC-AUC = %.4f. Van tiep tuc...",
                    best_val_so_far[0],
                )
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = scale_pos_weight

    print()
    print("=" * 60)
    print(" OPTUNA TUNING KET QUA")
    print("=" * 60)
    print(f"  So trials     : {n_trials}")
    print(f"  Best ROC-AUC  : {study.best_value:.4f}  (tren validation set)")
    print("  Best params   :")
    param_display_order = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda",
    ]
    for k in param_display_order:
        v = best_params.get(k, study.best_params.get(k))
        if v is None:
            continue
        if isinstance(v, float):
            print(f"    {k:<20} : {v:.6f}")
        else:
            print(f"    {k:<20} : {v}")
    print("=" * 60)
    print()

    return best_params


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — TRAIN MODEL CUỐI
# ═════════════════════════════════════════════════════════════════════════════

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: dict,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """
    Re-train on the combined train+val split using the best hyperparameters.

    Merging validation data back into training allows the final model to see
    15 % more labelled examples.  The test set is untouched throughout and
    acts as a genuinely held-out evaluation set.

    Parameters
    ----------
    X_train, y_train : training split.
    X_val, y_val     : validation split — concatenated to X_train before fit.
    best_params      : dict from tune_hyperparameters().
    random_state     : seed. Default 42.

    Returns
    -------
    xgb.XGBClassifier
        Trained final model.
    """
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    logger.info(
        "Training model cuoi tren %d samples (%d viral)...",
        len(X_trainval), int(y_trainval.sum()),
    )
    print(f"  Training model cuoi tren {len(X_trainval):,} samples...")

    # Remove keys not valid as constructor params for the final fit
    final_keys_exclude = {"early_stopping_rounds"}
    params = {
        k: v for k, v in best_params.items() if k not in final_keys_exclude
    }
    params.update({
        "eval_metric":  "auc",
        "random_state": random_state,
        "tree_method":  "hist",
    })

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_trainval, y_trainval, verbose=False)

    logger.info("Final model training hoàn tất.")
    return final_model


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — ĐÁNH GIÁ MODEL
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate the trained model on the held-out test set and persist charts.

    Produces and saves a 2×2 diagnostic figure to ``plots/evaluation.png``.

    Parameters
    ----------
    model     : trained XGBClassifier.
    X_test    : test features.
    y_test    : test labels.
    threshold : classification threshold for binary predictions. Default 0.5.

    Returns
    -------
    dict
        Keys: roc_auc, pr_auc, f1, precision, recall, optimal_threshold.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc   = roc_auc_score(y_test, y_prob)
    pr_auc    = average_precision_score(y_test, y_prob)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)

    # Optimal threshold from PR curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx          = int(np.argmax(f1_scores[:-1]))  # last element has no threshold
    optimal_threshold = float(thresholds[best_idx])
    best_f1           = float(f1_scores[best_idx])

    # Threshold recommendations
    lower_threshold  = round(max(0.1, optimal_threshold - 0.10), 2)
    higher_threshold = round(min(0.9, optimal_threshold + 0.10), 2)

    # ROC-AUC label
    if roc_auc >= 0.88:
        auc_label = "Rat tot"
    elif roc_auc >= 0.82:
        auc_label = "Tot"
    elif roc_auc >= 0.75:
        auc_label = "Chap nhan duoc"
    else:
        auc_label = "Can xem xet lai features hoac label"

    print()
    print("=" * 60)
    print(" KET QUA DANH GIA MODEL (Test Set)")
    print("=" * 60)
    print(f"  ROC-AUC   : {roc_auc:.4f}   [{auc_label}]")
    print(f"  PR-AUC    : {pr_auc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Threshold : {threshold}  (mac dinh)")
    print("-" * 60)
    print(f"  Threshold toi uu (theo F1) : {optimal_threshold:.3f}")
    print(f"  F1 tai threshold toi uu    : {best_f1:.4f}")
    print("-" * 60)
    print("  Khuyen nghi:")
    print(f"    Neu uu tien khong bo sot viral (recall cao):")
    print(f"      Dung threshold = {lower_threshold:.2f}")
    print(f"    Neu uu tien chinh xac khi predict viral (precision cao):")
    print(f"      Dung threshold = {higher_threshold:.2f}")
    print("=" * 60)
    print()

    # ── Plots ──────────────────────────────────────────────────────────────
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Evaluation — YouTube Viral Prediction", fontsize=14, fontweight="bold")

    # 1. ROC Curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    # 2. Precision-Recall Curve
    ax = axes[0, 1]
    ax.plot(recalls[:-1], precisions[:-1], color="#DD8452", lw=2,
            label=f"PR (AUC = {pr_auc:.4f})")
    ax.axvline(recalls[best_idx], color="#C44E52", linestyle="--", lw=1.5,
               label=f"Optimal threshold = {optimal_threshold:.3f}")
    ax.scatter([recalls[best_idx]], [precisions[best_idx]],
               color="#C44E52", s=80, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # 3. Confusion Matrix
    ax = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["Not Viral", "Viral"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix  (threshold={threshold})")

    # 4. Feature Importance top 20
    ax = axes[1, 1]
    fi = model.get_booster().get_score(importance_type="gain")
    fi_series = pd.Series(fi).sort_values(ascending=False).head(20)
    colors = ["#4C72B0"] * len(fi_series)
    ax.barh(range(len(fi_series)), fi_series.values[::-1], color=colors[::-1], edgecolor="white")
    ax.set_yticks(range(len(fi_series)))
    ax.set_yticklabels(fi_series.index[::-1], fontsize=8)
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance (Top 20, by Gain)")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plot_path = PLOTS_DIR / "evaluation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Evaluation plots saved → %s", plot_path)
    print(f"  Bieu do da luu tai: {plot_path}")
    print()

    return {
        "roc_auc":           round(roc_auc, 6),
        "pr_auc":            round(pr_auc, 6),
        "f1":                round(f1, 6),
        "precision":         round(precision, 6),
        "recall":            round(recall, 6),
        "optimal_threshold": round(optimal_threshold, 6),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — PIPELINE TỔNG
# ═════════════════════════════════════════════════════════════════════════════

def run_training_pipeline(
    data: dict,
    n_trials: int = 50,
    random_state: int = 42,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    End-to-end training pipeline: label → features → split → tune → train → evaluate.

    Parameters
    ----------
    data : dict
        Output of load_all_intermediate_data().
        Expected keys: "videos", "engagement", "channels".
    n_trials : int, optional
        Number of Optuna trials. Default 50.
    random_state : int, optional
        Global random seed. Default 42.

    Returns
    -------
    final_model : xgb.XGBClassifier
    training_config : dict
        Contains feature_list, train_medians, label_encoders,
        viral_threshold, optimal_threshold, viral_rate_train,
        scale_pos_weight, trained_at, model_version, metrics.
    """
    from ml.features import (
        FEATURE_COLS,
        engineer_features,
        fill_missing,
        merge_all_sources,
    )
    from ml.label import define_viral_label

    np.random.seed(random_state)

    # ── Step 1 — Label ───────────────────────────────────────────────────────
    logger.info("STEP 1/8  Define viral label …")
    df_labeled = define_viral_label(
        df_engagement=data["engagement"],
        df_channels=data["channels"],
        strategy="auto",
    )

    # ── Step 2 — Merge ───────────────────────────────────────────────────────
    logger.info("STEP 2/8  Merge all sources …")
    df_merged = merge_all_sources(
        df_labeled=df_labeled,
        df_videos=data["videos"],
        df_engagement=data["engagement"],
        df_channels=data["channels"],
    )

    # ── Step 3 — Feature engineering ─────────────────────────────────────────
    logger.info("STEP 3/8  Engineer features …")
    X_raw, label_encoders = engineer_features(df_merged)

    # ── Step 4 — Fill missing ─────────────────────────────────────────────────
    logger.info("STEP 4/8  Fill missing values …")
    X, train_medians = fill_missing(X_raw)
    y = df_merged["is_viral"].reset_index(drop=True)

    # ── Step 5 — Time-based split ─────────────────────────────────────────────
    logger.info("STEP 5/8  Time-based split …")
    published_at = df_merged["published_at"].reset_index(drop=True)
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(
        X, y, published_at,
    )

    # ── Step 6 — Optuna tuning ────────────────────────────────────────────────
    logger.info("STEP 6/8  Optuna hyperparameter tuning (%d trials) …", n_trials)
    best_params = tune_hyperparameters(
        X_train, y_train, X_val, y_val,
        n_trials=n_trials, random_state=random_state,
    )

    # ── Step 7 — Train final model ────────────────────────────────────────────
    logger.info("STEP 7/8  Train final model …")
    final_model = train_final_model(
        X_train, y_train, X_val, y_val,
        best_params=best_params, random_state=random_state,
    )

    # ── Step 8 — Evaluate ─────────────────────────────────────────────────────
    logger.info("STEP 8/8  Evaluate on test set …")
    metrics = evaluate_model(final_model, X_test, y_test, threshold=0.5)

    # ── Package config ────────────────────────────────────────────────────────
    scale_pos_weight = best_params.get(
        "scale_pos_weight",
        int((y_train == 0).sum()) / max(1, int((y_train == 1).sum())),
    )
    training_config = {
        "feature_list":       list(X.columns),
        "train_medians":      train_medians,
        "label_encoders":     label_encoders,
        "viral_threshold":    0.5,
        "optimal_threshold":  metrics["optimal_threshold"],
        "viral_rate_train":   round(float(y_train.mean() * 100), 2),
        "scale_pos_weight":   round(float(scale_pos_weight), 4),
        "trained_at":         datetime.now(timezone.utc).isoformat(),
        "model_version":      "v1",
        "metrics":            metrics,
    }

    logger.info(
        "run_training_pipeline hoàn tất  |  ROC-AUC=%.4f  F1=%.4f  Optimal-thr=%.3f",
        metrics["roc_auc"], metrics["f1"], metrics["optimal_threshold"],
    )
    return final_model, training_config


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 6 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from ml.data_loader import load_all_intermediate_data
    from ml.save_load import save_model

    print("\n>>> BƯỚC 1: Load dữ liệu từ BigQuery …")
    data = load_all_intermediate_data()

    print("\n>>> BƯỚC 2-8: Run training pipeline …")
    model, config = run_training_pipeline(data, n_trials=50)

    print("\n>>> BƯỚC 9: Lưu model …")
    save_model(model, config, version="v1", overwrite=True)
    print("  Model da duoc luu tai models/xgb_viral_v1.json + feature_config_v1.yaml")
