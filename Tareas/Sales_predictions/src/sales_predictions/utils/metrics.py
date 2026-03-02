"""Metrics utilities."""

from __future__ import annotations

import math
from typing import Iterable


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    true_list = list(y_true)
    pred_list = list(y_pred)
    if len(true_list) != len(pred_list):
        raise ValueError("y_true and y_pred must have the same length")
    if not true_list:
        raise ValueError("y_true and y_pred must be non-empty")
    mse = sum((t - p) ** 2 for t, p in zip(true_list, pred_list)) / len(true_list)
    return math.sqrt(mse)
