
---

# Model

The model implemented is a **Ridge Regression model** trained on monthly aggregated sales data.

Training output example:

```
Modelo entrenado - RMSE: 2.548426
```

---

# Docker Containers

Two Docker images were built for SageMaker:

### Training Image

Responsible for running model training.

```
Dockerfile.train
ENTRYPOINT: train_sagemaker.py
```

### Inference Image

Responsible for serving predictions through a SageMaker endpoint.

```
Dockerfile.infer
ENTRYPOINT: sagemaker_inference
```

---

# Build Docker Images

```
docker build --network sagemaker -f sagemaker/Dockerfile.train -t sales-preds-train .
docker build --network sagemaker -f sagemaker/Dockerfile.infer -t sales-preds-infer .
```

---

# Push Images to AWS ECR

Repositories created:

```
sales-preds-train
sales-preds-infer
```

Images pushed to:

```
448591726855.dkr.ecr.us-east-1.amazonaws.com/sales-preds-train
448591726855.dkr.ecr.us-east-1.amazonaws.com/sales-preds-infer
```

---

# SageMaker Training Job

The training job was executed in SageMaker using the custom training container.

Training process:

1. Load dataset
2. Train Ridge regression
3. Evaluate RMSE
4. Save model artifact

Example log output:

```
action=train fit status=success algo=ridge
Modelo entrenado - RMSE: 2.548426
model_path=/opt/ml/model/model.joblib
```

Training completed successfully.

---

# SageMaker Real-Time Endpoint

After training, the inference container was deployed as a **real-time endpoint**.

Endpoint name:

```
sales-preds-realtime-v2
```

Endpoint status:

```
InService
```

---

# Real-Time Prediction

Example request:

```python
sample_payload = [
{
"date_block_num": 34,
"shop_id": 31,
"item_id": 5560,
"item_category_id": 37
}
]