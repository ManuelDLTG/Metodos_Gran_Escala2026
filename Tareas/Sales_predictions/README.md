# Tarea 07 — SageMaker Pipelines (BYOC End-to-End)

## Overview
This project implements an end-to-end Machine Learning pipeline using **Amazon SageMaker Pipelines** with fully custom containers (BYOC).

The pipeline automates:
- Data preprocessing
- Model training
- Model evaluation (RMSE)
- Conditional logic
- Model registration
- Batch inference

---

## Architecture

Processing → Training → Evaluation → Condition  
                              ↓  
                    Model → Transform → Register  
                              ↓  
                            Fail  

---

## Pipeline Steps

### 1. ProcessingStep — Preprocessing
- Container: `sales-preds-processing`
- Script: `preprocess.py`
- Outputs:
  - train_split.csv
  - validation_split.csv
  - test_features.csv

---

### 2. TrainingStep — Training
- Container: `sales-preds-train`
- Script: `train_pipeline.py`
- Model: Ridge Regression
- Output:
  - model.joblib

---

### 3. ProcessingStep — Evaluation
- Script: `evaluate.py`
- Metric: RMSE
- Output:
  - evaluation.json

Example:
{
  "regression_metrics": {
    "rmse": {
      "value": 2.54
    }
  }
}

---

### 4. ConditionStep
If RMSE ≤ threshold:
- Register model
- Run batch transform

Else:
- Fail pipeline

---

### 5. ModelStep — Create Model
- Container: `sales-preds-infer`

---

### 6. TransformStep — Batch Inference
- Generates predictions
- Stores results in S3

---

### 7. RegisterModel
- Registers model in SageMaker Model Registry

---

## AWS Components Used
- SageMaker Pipelines
- ECR (custom containers)
- S3 (artifacts)
- CloudWatch (logs)

---

## Repository Structure
- processing/
- sagemaker/
- notebooks/
- docs/screenshots/

---

## Evidence (Screenshots Required)
You must include:
1. Pipeline execution (Succeeded)
2. Pipeline graph
3. ECR images
4. Processing outputs
5. Evaluation output
6. Model registry
7. Batch transform output

---

![Pylint 10/10](docs/1_pipeline.png)


![Pylint 10/10](docs/2_pipeline.png)

---

## Conclusion
This project demonstrates a production-ready ML pipeline using custom containers, enabling full control over training and inference workflows.

---

## Author
Manuel De la Tejera
MSc Data Science — ITAM
CFA Level I Passed
