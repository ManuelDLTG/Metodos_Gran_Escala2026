"""Inference / submission generation."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sales_predictions.utils.data_validation import ensure_file_exists
from sales_predictions.utils.logging import get_logger


@dataclass(frozen=True)
class InferenceResult:
    """Inference result summary."""

    submission_path: Path
    rows: int


def run_inference(
    inference_path: Path,
    model_path: Path,
    pred_out: Path,
    clip_min: float,
    clip_max: float,
) -> InferenceResult:
    """Run inference and write a Kaggle submission CSV."""
    logger = get_logger("inference")
    start_time = time.time()

    ensure_file_exists(inference_path, hint="Run: python scripts/prep.py")
    ensure_file_exists(model_path, hint="Run: python scripts/train.py")

    logger.info("action=inference load_features status=started path=%s", str(inference_path))
    df = pd.read_csv(inference_path, compression="gzip")
    logger.info(
        "action=inference load_features status=success rows=%d cols=%d", len(df), df.shape[1]
    )

    payload = joblib.load(model_path)
    model = payload["model"]
    feat_cols = payload["feature_columns"]

    missing = [col for col in feat_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for inference: {missing}")
    if "ID" not in df.columns:
        raise ValueError("Missing column 'ID' (from Kaggle test.csv).")

    preds = model.predict(df[feat_cols])
    preds = np.clip(preds, clip_min, clip_max)

    pred_out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"ID": df["ID"], "item_cnt_month": preds})
    out.to_csv(pred_out, index=False)

    duration = time.time() - start_time
    logger.info(
        "Predicciones guardadas: %d registros | clip_min=%.2f clip_max=%.2f",
        len(out),
        clip_min,
        clip_max,
    )
    logger.info("Tiempo de ejecución: %.2f segundos", duration)
    logger.info("action=inference write_submission status=success path=%s", str(pred_out))

    return InferenceResult(submission_path=pred_out, rows=len(out))


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inference-path", type=Path, default=Path("data/inference/test_features.csv.gz")
    )
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--pred-out", type=Path, default=Path("data/predictions/submission.csv"))
    parser.add_argument("--clip-min", type=float, default=0.0)
    parser.add_argument("--clip-max", type=float, default=20.0)
    return parser


def main() -> None:
    """CLI main."""
    args = build_argparser().parse_args()
    logger = get_logger("inference")

    logger.info("action=inference status=started")
    try:
        res = run_inference(
            inference_path=args.inference_path,
            model_path=args.model_path,
            pred_out=args.pred_out,
            clip_min=args.clip_min,
            clip_max=args.clip_max,
        )
        logger.info("action=inference status=success submission_path=%s", str(res.submission_path))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("action=inference status=failure error=%s", str(exc))
        raise


if __name__ == "__main__":
    main()
