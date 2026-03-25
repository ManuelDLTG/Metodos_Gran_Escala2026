"""SageMaker training entrypoint for pipeline mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sales_predictions.train import pick_model
from sales_predictions.utils.logging import get_logger


TRAIN_DIR = Path("/opt/ml/input/data/train")
VALIDATION_DIR = Path("/opt/ml/input/data/validation")
MODEL_DIR = Path("/opt/ml/model")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline training entrypoint")
    parser.add_argument("--train-file", type=str, default="train_split.csv")
    parser.add_argument("--validation-file", type=str, default="validation_split.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--algo",
        type=str,
        default="ridge",
        choices=["auto", "xgboost", "lightgbm", "ridge"],
    )
    parser.add_argument(
        "--features",
        type=str,
        default="date_block_num,shop_id,item_id,item_category_id",
    )
    parser.add_argument("--model-name", type=str, default="model.joblib")
    return parser


def main() -> None:
    logger = get_logger("train_pipeline")
    args, unknown = build_argparser().parse_known_args()
    logger.info("unknown_args=%s", unknown)

    train_path = TRAIN_DIR / args.train_file
    validation_path = VALIDATION_DIR / args.validation_file
    model_out = MODEL_DIR / args.model_name

    feature_columns = [col.strip() for col in args.features.split(",") if col.strip()]

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(validation_path)

    x_train = df_train[feature_columns]
    y_train = df_train["item_cnt_month"].astype(float)

    x_val = df_val[feature_columns]
    y_val = df_val["item_cnt_month"].astype(float)

    model, algo_used = pick_model(args.algo, random_state=args.seed)

    logger.info("action=train_pipeline fit status=started algo=%s", algo_used)

    if algo_used in {"xgboost", "lightgbm"}:
        try:
            model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        except TypeError:
            model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train)

    preds = model.predict(x_val)
    rmse = float(np.sqrt(((y_val.to_numpy() - preds) ** 2).mean()))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "algo": algo_used,
        "validation_rmse": rmse,
    }

    joblib.dump(payload, model_out)

    logger.info("action=train_pipeline status=success algo=%s rmse=%.6f", algo_used, rmse)
    logger.info("model saved at %s", str(model_out))


if __name__ == "__main__":
    main()