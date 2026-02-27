"""Data preparation (ETL).

Transforms the raw Kaggle inputs into two compressed datasets:

1) Monthly training dataset (for supervised learning)
2) Feature table for inference on the Kaggle test set

Outputs
-------
- data/prep/dataset_monthly.csv.gz
- data/inference/test_features.csv.gz

Logging
-------
Logs are written to `artifacts/logs/`.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from sales_predictions.utils.data_validation import ensure_file_exists
from sales_predictions.utils.logging import get_logger


RUTA_DATOS_RAW = Path("data/raw")
RUTA_DATOS_PREP = Path("data/prep")
RUTA_DATOS_INFERENCE = Path("data/inference")
NOMBRE_DATASET_MENSUAL = "dataset_monthly.csv.gz"
NOMBRE_FEATURES_TEST = "test_features.csv.gz"
CLIP_MIN = 0
CLIP_MAX = 20


def _load_raw_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw Kaggle tables."""
    sales_train_path = raw_dir / "sales_train.csv"
    test_path = raw_dir / "test.csv"
    items_path = raw_dir / "items_en.csv"

    ensure_file_exists(sales_train_path, hint=f"Check that {raw_dir}/ contains Kaggle files")
    ensure_file_exists(test_path, hint=f"Check that {raw_dir}/ contains Kaggle files")
    ensure_file_exists(items_path, hint=f"Check that {raw_dir}/ contains Kaggle files")

    df_sales = pd.read_csv(sales_train_path)
    df_test = pd.read_csv(test_path)
    df_items = pd.read_csv(items_path)

    return df_sales, df_test, df_items


def _build_monthly_dataset(df_sales: pd.DataFrame, df_items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales into monthly targets and optionally merge item categories."""
    df_monthly = (
        df_sales.groupby(["date_block_num", "shop_id", "item_id"], as_index=False)["item_cnt_day"]
        .sum()
        .rename(columns={"item_cnt_day": "item_cnt_month"})
    )

    df_monthly["item_cnt_month"] = df_monthly["item_cnt_month"].clip(CLIP_MIN, CLIP_MAX)

    if "item_category_id" in df_items.columns:
        df_monthly = df_monthly.merge(
            df_items[["item_id", "item_category_id"]], on="item_id", how="left"
        )

    return df_monthly


def _build_inference_features(
    df_test: pd.DataFrame, df_items: pd.DataFrame, last_block: int
) -> pd.DataFrame:
    """Create features for inference by adding the next date_block_num and merging categories."""
    df_inf = df_test.copy()
    df_inf["date_block_num"] = last_block + 1

    if "item_category_id" in df_items.columns:
        df_inf = df_inf.merge(df_items[["item_id", "item_category_id"]], on="item_id", how="left")

    return df_inf


def prep_datasets(
    raw_dir: Path,
    prep_dir: Path,
    inference_dir: Path,
    prep_name: str,
    inference_name: str,
) -> tuple[Path, Path]:
    """Generate training and inference datasets.

    Parameters
    ----------
    raw_dir:
        Directory containing Kaggle raw CSVs.
    prep_dir:
        Directory where the monthly training dataset will be written.
    inference_dir:
        Directory where the inference feature table will be written.
    prep_name:
        Filename for the monthly dataset.
    inference_name:
        Filename for the inference table.

    Returns
    -------
    tuple[Path, Path]
        Paths of (monthly_dataset_path, inference_features_path).
    """
    logger = get_logger("prep")
    start_time = time.time()

    prep_dir.mkdir(parents=True, exist_ok=True)
    inference_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Iniciando preprocesamiento de datos...")
    df_sales, df_test, df_items = _load_raw_tables(raw_dir)
    logger.info(
        "Datos cargados: sales=%d, test=%d, items=%d",
        len(df_sales),
        len(df_test),
        len(df_items),
    )

    df_monthly = _build_monthly_dataset(df_sales, df_items)
    out_prep = prep_dir / prep_name
    df_monthly.to_csv(out_prep, index=False, compression="gzip")
    logger.info("Dataset mensual guardado: %s (rows=%d)", str(out_prep), len(df_monthly))

    last_block = int(df_monthly["date_block_num"].max())
    df_inf = _build_inference_features(df_test, df_items, last_block)
    out_inf = inference_dir / inference_name
    df_inf.to_csv(out_inf, index=False, compression="gzip")
    logger.info("Features de inferencia guardadas: %s (rows=%d)", str(out_inf), len(df_inf))

    duration = time.time() - start_time
    logger.info("Tiempo de ejecución: %.2f segundos", duration)

    return out_prep, out_inf


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RUTA_DATOS_RAW)
    parser.add_argument("--prep-dir", type=Path, default=RUTA_DATOS_PREP)
    parser.add_argument("--inference-dir", type=Path, default=RUTA_DATOS_INFERENCE)
    parser.add_argument("--prep-name", type=str, default=NOMBRE_DATASET_MENSUAL)
    parser.add_argument("--inference-name", type=str, default=NOMBRE_FEATURES_TEST)
    return parser


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    logger = get_logger("prep")

    logger.info("action=prep status=started")
    try:
        out_prep, out_inf = prep_datasets(
            raw_dir=args.raw_dir,
            prep_dir=args.prep_dir,
            inference_dir=args.inference_dir,
            prep_name=args.prep_name,
            inference_name=args.inference_name,
        )
        logger.info(
            "action=prep status=success train_path=%s inf_path=%s",
            str(out_prep),
            str(out_inf),
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("action=prep status=failure error=%s", str(exc))
        raise


if __name__ == "__main__":
    main()
