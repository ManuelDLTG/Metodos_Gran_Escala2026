# SageMaker Processing Job --- BYOC (Feature Engineering Pipeline)

## Overview

This project implements a **data preprocessing pipeline in Amazon
SageMaker using a Bring Your Own Container (BYOC)** architecture.

The objective of this stage is to transform **raw sales data into
structured feature datasets** that can later be used for machine
learning training and inference.

Instead of preprocessing data locally, the pipeline runs inside a
**SageMaker Processing Job**, ensuring:

-   reproducibility
-   scalability
-   cloud‑native execution
-   separation between compute and storage

------------------------------------------------------------------------

# Pipeline Architecture

S3 Raw Data\
↓\
SageMaker Processing Job (BYOC container)\
↓\
Feature Engineering (`processing/preprocess.py`)\
↓\
Processed datasets stored in S3\
↓\
Training / Inference pipeline

------------------------------------------------------------------------

# Container Architecture (BYOC)

A custom Docker container was built for preprocessing.

The container includes:

-   Python
-   pandas
-   numpy
-   project source code

The container executes:

    processing/preprocess.py

The image is stored in **Amazon ECR**.

------------------------------------------------------------------------

# Processing Job Execution

The preprocessing job runs using the **SageMaker ScriptProcessor API**.

Input data is mounted inside the container at:

    /opt/ml/processing/input

Outputs are written to:

    /opt/ml/processing/output

------------------------------------------------------------------------

# Feature Engineering

The preprocessing pipeline performs:

-   merge between sales and item metadata
-   monthly aggregation of sales
-   generation of the modeling dataset
-   train / validation splits
-   inference feature construction

Generated features:

-   `date_block_num`
-   `shop_id`
-   `item_id`
-   `item_cnt_month`
-   `item_category_id`

------------------------------------------------------------------------

# Dataset Preview

![Dataset preview](docs/screenshots/02_dataset_preview.png)

Example rows from the aggregated monthly dataset.

------------------------------------------------------------------------

# Inference Dataset

![Inference preview](docs/screenshots/03_inference_preview.png)

Features prepared for inference on the Kaggle test dataset.

Dataset sizes produced by the processing job:

    full:        163525 rows
    train:       123159 rows
    validation:   40366 rows
    inference:   214200 rows

------------------------------------------------------------------------

# S3 Storage Structure

![S3 structure](docs/screenshots/01_processing_outputs.png)

Processed datasets are written to the S3 bucket using this structure:

    sales-predictions/
    ├── raw/
    ├── processed/
    ├── train/
    └── output/

------------------------------------------------------------------------

# ECR Container Image

![ECR image](docs/screenshots/04_ecr_image.png)

The preprocessing container is stored in **Amazon Elastic Container
Registry (ECR)**.

Repository:

    sales-preds-processing

------------------------------------------------------------------------

# SageMaker Processing Job

![Processing job](docs/screenshots/05_processing_job_completed.png)

The SageMaker processing job successfully completed and produced the
datasets used for model training.

------------------------------------------------------------------------

# Repository Structure

    Tareas/Sales_predictions/

    processing/
     ├── container/
     │     └── Dockerfile
     │
     └── preprocess.py

    notebooks/
     └── tarea06_sagemaker_processing_byoc.ipynb

    data/
     ├── raw/
     └── prep/

    docs/
     └── screenshots/

------------------------------------------------------------------------

# Technologies Used

-   Amazon SageMaker
-   Amazon S3
-   Amazon ECR
-   Docker
-   Python
-   pandas

------------------------------------------------------------------------

# Result

The preprocessing pipeline converts raw sales data into a structured
feature dataset ready for machine learning workflows.

This stage integrates with future steps such as:

-   model training
-   model deployment
-   real‑time inference endpoints
