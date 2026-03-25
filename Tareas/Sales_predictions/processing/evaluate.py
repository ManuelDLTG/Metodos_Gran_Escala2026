from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


MODEL_DIR = Path("/opt/ml/processing/input/model")
TEST_DIR = Path("/opt/ml/processing/input/test")
OUTPUT_DIR = Path("/opt/ml/processing/output/evaluation")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "model.joblib"
    test_path = TEST_DIR / "test_split.csv"

    payload = joblib.load(model_path)
    model = payload["model"]
    feature_columns = payload["feature_columns"]

    df_test = pd.read_csv(test_path)

    x_test = df_test[feature_columns]
    y_test = df_test["item_cnt_month"].astype(float)

    preds = model.predict(x_test)
    rmse = float(np.sqrt(((y_test.to_numpy() - preds) ** 2).mean()))

    evaluation = {
        "regression_metrics": {
            "rmse": {
                "value": rmse,
                "standard_deviation": 0.0
            }
        }
    }

    with open(OUTPUT_DIR / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f)

    print("Evaluation completed successfully")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()