from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from sales_predictions.prep import _build_inference_features, _build_monthly_dataset


INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")

TRAIN_OUT = OUTPUT_DIR / "train"
VALIDATION_OUT = OUTPUT_DIR / "validation"
FULL_OUT = OUTPUT_DIR / "full"
INFERENCE_OUT = OUTPUT_DIR / "inference"


def main() -> None:
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    VALIDATION_OUT.mkdir(parents=True, exist_ok=True)
    FULL_OUT.mkdir(parents=True, exist_ok=True)
    INFERENCE_OUT.mkdir(parents=True, exist_ok=True)

    sales_path = INPUT_DIR / "sales_train.csv"
    test_path = INPUT_DIR / "test.csv"
    items_path = INPUT_DIR / "items_en.csv"

    df_sales = pd.read_csv(sales_path)
    df_test = pd.read_csv(test_path)
    df_items = pd.read_csv(items_path)

    df_monthly = _build_monthly_dataset(df_sales, df_items)

    last_block = int(df_monthly["date_block_num"].max())
    val_block = last_block

    train_split = df_monthly[df_monthly["date_block_num"] < val_block].copy()
    validation_split = df_monthly[df_monthly["date_block_num"] == val_block].copy()

    df_inf = _build_inference_features(df_test, df_items, last_block)

    df_monthly.to_csv(FULL_OUT / "dataset_monthly.csv", index=False)
    train_split.to_csv(TRAIN_OUT / "train_split.csv", index=False)
    validation_split.to_csv(VALIDATION_OUT / "validation_split.csv", index=False)
    df_inf.to_csv(INFERENCE_OUT / "test_features.csv", index=False)

    print("Processing completed successfully")
    print(f"dataset_monthly rows: {len(df_monthly)}")
    print(f"train_split rows: {len(train_split)}")
    print(f"validation_split rows: {len(validation_split)}")
    print(f"test_features rows: {len(df_inf)}")


if __name__ == "__main__":
    main()