"""Model training.

Reads the prepared monthly dataset and trains a regression model.
The training payload is saved as a single artifact containing:

- fitted model
- feature columns used
- validation block
- algorithm identifier

Logging
-------
Logs are written to `artifacts/logs/`.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from sales_predictions.utils.data_validation import ensure_file_exists
from sales_predictions.utils.logging import get_logger

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass(frozen=True)
class TrainResult:
    """Training result summary."""

    rmse: float
    model_path: Path
    feature_columns: list[str]
    algo: str


def pick_model(algo: str, *, random_state: int) -> tuple[object, str]:
    """Pick and initialize a model."""
    algo_lower = algo.lower()

    if algo_lower == "auto":
        if HAS_XGB:
            algo_lower = "xgboost"
        elif HAS_LGB:
            algo_lower = "lightgbm"
        else:
            algo_lower = "ridge"

    if algo_lower == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost is not installed; use --algo ridge or install xgboost")
        model = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        return model, "xgboost"

    if algo_lower == "lightgbm":
        if not HAS_LGB:
            raise RuntimeError("lightgbm is not installed; use --algo ridge or install lightgbm")
        model = lgb.LGBMRegressor(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
        )
        return model, "lightgbm"

    if algo_lower == "ridge":
        return Ridge(alpha=1.0, random_state=random_state), "ridge"

    raise ValueError("Invalid algo. Use: auto | xgboost | lightgbm | ridge")


def train_model(
    prep_path: Path,
    model_out: Path,
    val_block: int,
    configured_features: list[str],
    algo: str,
    seed: int,
) -> TrainResult:
    """Train a model and persist the training payload."""
    logger = get_logger("train")
    start_time = time.time()

    ensure_file_exists(prep_path, hint="Run: python scripts/prep.py")

    logger.info("action=train load_prep status=started path=%s", str(prep_path))
    df = pd.read_csv(prep_path, compression="gzip")
    logger.info("action=train load_prep status=success rows=%d cols=%d", len(df), df.shape[1])

    feature_columns = [col for col in configured_features if col in df.columns]
    required = {"date_block_num", "shop_id", "item_id"}
    if not required.issubset(set(feature_columns)):
        raise ValueError(f"Required features: {sorted(required)}. Got: {feature_columns}")

    train_df = df[df["date_block_num"] < val_block].copy()
    val_df = df[df["date_block_num"] == val_block].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Empty train/val split. Check val_block or data.")

    x_train = train_df[feature_columns]
    y_train = train_df["item_cnt_month"].astype(float)
    x_val = val_df[feature_columns]
    y_val = val_df["item_cnt_month"].astype(float)

    logger.info(
        "action=train split status=success train_rows=%d val_rows=%d n_features=%d val_block=%d",
        len(train_df),
        len(val_df),
        len(feature_columns),
        val_block,
    )

    model, algo_used = pick_model(algo, random_state=seed)
    logger.info("action=train fit status=started algo=%s", algo_used)

    if algo_used in {"xgboost", "lightgbm"}:
        try:
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)

    logger.info("action=train fit status=success algo=%s", algo_used)

    preds = model.predict(x_val)
    rmse_value = float(np.sqrt(((y_val.to_numpy() - preds) ** 2).mean()))

    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "val_block": val_block,
        "algo": algo_used,
    }
    joblib.dump(payload, model_out)

    duration = time.time() - start_time
    logger.info("Modelo entrenado - RMSE: %.6f", rmse_value)
    logger.info("Tiempo de ejecución: %.2f segundos", duration)
    logger.info("action=train save status=success model_path=%s", str(model_out))

    return TrainResult(
        rmse=rmse_value,
        model_path=model_out,
        feature_columns=feature_columns,
        algo=algo_used,
    )


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-path", type=Path, default=Path("data/prep/dataset_monthly.csv.gz"))
    parser.add_argument("--model-out", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--val-block", type=int, default=33)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algo", type=str, default="auto", choices=["auto", "xgboost", "lightgbm", "ridge"]
    )
    parser.add_argument(
        "--features",
        type=str,
        default="date_block_num,shop_id,item_id,item_category_id",
        help="Comma-separated list.",
    )
    return parser


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    logger = get_logger("train")

    configured_features = [col.strip() for col in args.features.split(",") if col.strip()]

    logger.info("action=train status=started")
    try:
        res = train_model(
            prep_path=args.prep_path,
            model_out=args.model_out,
            val_block=args.val_block,
            configured_features=configured_features,
            algo=args.algo,
            seed=args.seed,
        )
        logger.info(
            "action=train status=success algo=%s rmse=%.6f model_path=%s",
            res.algo,
            res.rmse,
            str(res.model_path),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("action=train status=failure error=%s", str(exc))
        raise


if __name__ == "__main__":
    main()
