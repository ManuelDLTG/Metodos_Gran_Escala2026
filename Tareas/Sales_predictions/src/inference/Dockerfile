FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    joblib==1.3.2 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.4.2 \
    PyYAML==6.0.1 \
    seaborn==0.13.2

COPY . /app
RUN pip install -e .

ENTRYPOINT ["python", "-m", "inference"]