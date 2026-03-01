"""
Model persistence for the YouTube viral prediction pipeline.

Saves and loads the XGBoost model (JSON format) and its associated
training configuration (YAML) to / from the models/ directory.

Pipeline order:1
    ml/train.py  → save_model()  →  models/xgb_viral_{v}.json
                                    models/feature_config_{v}.yaml
    ml/predict.py → load_model() ←  same files

Usage:
    from ml.save_load import save_model, load_model, list_saved_models

    save_model(model, config, version="v1")
    model, config = load_model(version="v1")
    list_saved_models()

Run standalone:
    python -m ml.save_load
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
import yaml
from sklearn.preprocessing import LabelEncoder

# ── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── PHẦN 1 — CẤU HÌNH ĐƯỜNG DẪN ─────────────────────────────────────────────
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _model_path(version: str) -> Path:
    """Return the XGBoost model file path for a given version."""
    return MODELS_DIR / f"xgb_viral_{version}.json"


def _config_path(version: str) -> Path:
    """Return the YAML config file path for a given version."""
    return MODELS_DIR / f"feature_config_{version}.yaml"


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 2 — HÀM LƯU MODEL
# ═════════════════════════════════════════════════════════════════════════════

def save_model(
    model: xgb.XGBClassifier,
    config: dict,
    version: str = "v1",
    overwrite: bool = False,
) -> None:
    """
    Persist the trained XGBoost model and its full training config to disk.

    Files written:
        models/xgb_viral_{version}.json        ← XGBoost native format
        models/feature_config_{version}.yaml   ← training config (YAML, UTF-8)

    Parameters
    ----------
    model     : Trained XGBClassifier.
    config    : dict returned by ``run_training_pipeline()``.  Must contain:
                feature_list, train_medians, label_encoders,
                viral_threshold, optimal_threshold,
                viral_rate_train, scale_pos_weight,
                trained_at, model_version, metrics.
    version   : Version tag appended to filenames, default ``"v1"``.
    overwrite : If ``False`` and files already exist, raises
                ``FileExistsError``.  Set to ``True`` to overwrite.

    Raises
    ------
    FileExistsError
        If model file already exists and ``overwrite=False``.
    """
    # BƯỚC 1 — Kiểm tra overwrite
    mpath = _model_path(version)
    if mpath.exists() and not overwrite:
        raise FileExistsError(
            f"Model {version} da ton tai tai {mpath}.\n"
            "Dung overwrite=True de ghi de, hoac doi sang version moi."
        )

    # BƯỚC 2 — Lưu model (XGBoost native JSON)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(mpath))
    logger.info("Model saved → %s", mpath)

    # BƯỚC 3 — Chuẩn bị config để lưu YAML (serialize numpy & sklearn objects)
    label_encoders_serializable = {}
    for col, encoder in config.get("label_encoders", {}).items():
        label_encoders_serializable[col] = {
            "classes": encoder.classes_.tolist(),
            "mapping": {str(c): int(i) for i, c in enumerate(encoder.classes_)},
        }

    # BƯỚC 4 — Lưu config sang YAML
    config_to_save = {
        "model_version":     version,
        "trained_at":        config["trained_at"],
        "feature_list":      config["feature_list"],
        "train_medians":     {k: float(v) for k, v in config.get("train_medians", {}).items()},
        "label_encoders":    label_encoders_serializable,
        "viral_threshold":   float(config["viral_threshold"]),
        "optimal_threshold": float(config["optimal_threshold"]),
        "viral_rate_train":  float(config["viral_rate_train"]),
        "scale_pos_weight":  float(config["scale_pos_weight"]),
        "metrics": {
            k: float(v) for k, v in config.get("metrics", {}).items()
        },
    }

    cpath = _config_path(version)
    with open(cpath, "w", encoding="utf-8") as f:
        yaml.dump(config_to_save, f, allow_unicode=True, sort_keys=False)
    logger.info("Config saved → %s", cpath)

    # BƯỚC 5 — Xác nhận đã lưu
    sep = "=" * 60
    dash = "-" * 60
    metrics = config.get("metrics", {})
    print(f"\n{sep}")
    print("  MODEL DA DUOC LUU")
    print(sep)
    print(f"  Model file  : {mpath}")
    print(f"  Config file : {cpath}")
    print(f"  Version     : {version}")
    print(f"  Trained at  : {config['trained_at']}")
    print(f"  ROC-AUC     : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  Viral rate  : {config['viral_rate_train']:.1f}%")
    print(f"  Features    : {len(config['feature_list'])} features")
    print(f"{sep}\n")


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 3 — HÀM LOAD MODEL
# ═════════════════════════════════════════════════════════════════════════════

def load_model(
    version: str = "v1",
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Load a saved XGBoost model and its YAML training config.

    Parameters
    ----------
    version : Version tag used when saving, default ``"v1"``.

    Returns
    -------
    (model, config) where config is a plain dict with LabelEncoders
    reconstructed from the YAML mapping.

    Raises
    ------
    FileNotFoundError
        If the model or config file is missing.
    ValueError
        If the YAML config file is corrupt or unreadable.
    """
    mpath = _model_path(version)
    cpath = _config_path(version)

    # BƯỚC 1 — Kiểm tra file tồn tại
    if not mpath.exists():
        raise FileNotFoundError(
            f"Khong tim thay model {version} tai {mpath}.\n"
            "Chay 'python -m ml.train' de train model truoc."
        )
    if not cpath.exists():
        # Check if old-style .pkl config exists to give a helpful hint
        old_pkl = MODELS_DIR / f"xgb_viral_{version}_config.pkl"
        hint = (
            f"  (Phat hien config cu dang .pkl tai {old_pkl} — "
            "hay train lai de tao file YAML moi.)"
            if old_pkl.exists()
            else ""
        )
        raise FileNotFoundError(
            f"Khong tim thay config {version} tai {cpath}.\n"
            f"Xoa model file va train lai.{hint}"
        )

    # BƯỚC 2 — Load model
    model = xgb.XGBClassifier()
    model.load_model(str(mpath))
    logger.info("Model loaded ← %s", mpath)

    # BƯỚC 3 — Load config từ YAML
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            config_raw: dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"File config YAML bi loi: {cpath}\n"
            "Hay xoa ca model va config roi chay 'python -m ml.train' lai."
        ) from exc

    if not isinstance(config_raw, dict):
        raise ValueError(
            f"Config YAML khong hop le (expected dict, got {type(config_raw)}): {cpath}"
        )

    # BƯỚC 4 — Tái tạo LabelEncoder từ mapping đã lưu
    label_encoders: dict[str, LabelEncoder] = {}
    for col, enc_data in config_raw.get("label_encoders", {}).items():
        le = LabelEncoder()
        le.classes_ = np.array(enc_data["classes"])
        label_encoders[col] = le
    config_raw["label_encoders"] = label_encoders

    # BƯỚC 5 — In thông tin
    metrics     = config_raw.get("metrics", {})
    trained_at  = config_raw.get("trained_at", "N/A")
    roc_auc     = metrics.get("roc_auc", float("nan"))
    n_features  = len(config_raw.get("feature_list", []))
    threshold   = config_raw.get("viral_threshold", 0.5)
    logger.info(
        "Model %s da duoc load. Trained at: %s | ROC-AUC: %.4f | "
        "Features: %d | Threshold: %.3f",
        version, trained_at, roc_auc, n_features, threshold,
    )

    return model, config_raw


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 4 — HÀM LIỆT KÊ VERSIONS
# ═════════════════════════════════════════════════════════════════════════════

def list_saved_models() -> list[dict]:
    """
    Scan ``models/`` and list every saved version.

    Returns
    -------
    list[dict]
        Each dict contains: version, trained_at, roc_auc,
        n_features, model_path, config_path.
    """
    results: list[dict] = []

    if not MODELS_DIR.exists():
        print("Thu muc models/ chua ton tai.")
        return results

    # Match all YAML configs; derive version from filename
    for cpath in sorted(MODELS_DIR.glob("feature_config_*.yaml")):
        version = cpath.stem.replace("feature_config_", "")
        mpath   = _model_path(version)

        entry: dict = {
            "version":     version,
            "trained_at":  "N/A",
            "roc_auc":     float("nan"),
            "n_features":  0,
            "model_path":  str(mpath),
            "config_path": str(cpath),
        }

        # Warn if model JSON is missing
        if not mpath.exists():
            logger.warning(
                "[CANH BAO] Config ton tai nhung model file bi mat: %s", mpath
            )

        # Read metadata from YAML (no need to reload model)
        try:
            with open(cpath, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict):
                metrics        = cfg.get("metrics", {})
                entry["trained_at"] = cfg.get("trained_at", "N/A")
                entry["roc_auc"]    = float(metrics.get("roc_auc", float("nan")))
                entry["n_features"] = len(cfg.get("feature_list", []))
        except (yaml.YAMLError, OSError):
            logger.warning("Khong doc duoc config: %s", cpath)

        results.append(entry)

    # ── Print table ───────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  DANH SACH MODEL DA LUU")
    print(sep)
    if not results:
        print("  (Chua co model nao duoc luu.)")
    else:
        header = f"  {'Version':<10}  {'Trained at':<22}  {'ROC-AUC':>7}  {'Features':>8}"
        divider = f"  {'--------':<10}  {'--------------------':<22}  {'-------':>7}  {'--------':>8}"
        print(header)
        print(divider)
        for r in results:
            roc = f"{r['roc_auc']:.4f}" if not np.isnan(r["roc_auc"]) else "  N/A  "
            print(
                f"  {r['version']:<10}  {str(r['trained_at']):<22}  {roc:>7}  {r['n_features']:>8}"
            )
        latest = results[-1]["version"]
        print(sep)
        print(f"  Dung load_model(\"{latest}\") de load model moi nhat.")
    print(f"{sep}\n")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 5 — HÀM XÓA MODEL CŨ
# ═════════════════════════════════════════════════════════════════════════════

def delete_model(version: str, confirm: bool = False) -> None:
    """
    Delete the model and config files for ``version``.

    Parameters
    ----------
    version : Version tag to delete.
    confirm : If ``True`` skip the interactive prompt and delete immediately.
              If ``False`` (default) ask the user to type ``y`` to confirm.
    """
    mpath = _model_path(version)
    cpath = _config_path(version)

    if not mpath.exists() and not cpath.exists():
        print(f"Khong tim thay any files for model version '{version}'.")
        return

    if not confirm:
        print(f"[CANH BAO] Ban sap xoa model '{version}':")
        if mpath.exists():
            print(f"  - {mpath}")
        if cpath.exists():
            print(f"  - {cpath}")
        answer = input("Ban co chac muon xoa model? (y/n): ").strip().lower()
        if answer != "y":
            print("Huy bo. Khong co gi bi xoa.")
            return

    for path in (mpath, cpath):
        if path.exists():
            path.unlink()
            logger.info("Da xoa: %s", path)

    print(f"Da xoa model {version}.")


# ═════════════════════════════════════════════════════════════════════════════
# PHẦN 6 — CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    list_saved_models()
