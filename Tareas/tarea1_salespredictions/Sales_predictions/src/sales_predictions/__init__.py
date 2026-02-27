"""Sales Predictions package.

Pipeline for Kaggle "Predict Future Sales":
- prep: builds monthly dataset + inference features
- train: trains a regressor and saves a model artifact
- inference: generates a Kaggle submission file
"""

from __future__ import annotations
