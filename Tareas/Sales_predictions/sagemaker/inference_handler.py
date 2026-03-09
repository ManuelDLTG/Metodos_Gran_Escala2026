"""SageMaker inference handler."""

from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


def model_fn(model_dir: str):
    model_path = Path(model_dir) / "model.joblib"
    return joblib.load(model_path)


def input_fn(input_data, content_type: str):
    if content_type == "application/json":
        body = json.loads(input_data)
        if isinstance(body, list):
            return pd.DataFrame(body)
        if isinstance(body, dict) and "instances" in body:
            return pd.DataFrame(body["instances"])
        raise ValueError("JSON inválido para inferencia.")
    elif content_type == "text/csv":
        return pd.read_csv(io.StringIO(input_data.decode("utf-8")))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(data, model):
    preds = model.predict(data)
    preds = np.clip(preds, 0, 20)
    return preds.tolist()


def output_fn(prediction, accept: str):
    if accept == "application/json":
        return json.dumps({"predictions": prediction}), accept
    raise ValueError(f"Unsupported accept type: {accept}")
