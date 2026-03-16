FROM python:3.11-slim

WORKDIR /opt/program

COPY pyproject.toml .
COPY src ./src
COPY processing/preprocess.py ./preprocess.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir pandas numpy scikit-learn
