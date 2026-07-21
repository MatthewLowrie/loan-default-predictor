# Loan Default Prediction API

An end-to-end, production-ready MLOps pipeline that generates synthetic financial data, trains multiple machine learning classifiers, tracks experiments, registers the best-performing model, and serves real-time predictions via a FastAPI REST API.

This project is fully automated and can be run step-by-step locally or packaged as a self-contained, platform-agnostic Docker container.

---

## Project Architecture

1. **Data Generation (`generate_data.py`)**: Produces 10,000 synthetic loan records with realistic credit risk correlations (distributions like log-normal income and normalized credit scores).
2. **Model Training (`train.py`)**: Trains baseline and class-weighted models (Logistic Regression & Random Forest) to handle class imbalance, logging metrics to an MLflow tracking SQLite database.
3. **Model Registration (`register_model.py`)**: Programmatically queries MLflow, selects the version with the highest ROC-AUC score, registers it under the model registry, and tags it with the `champion` alias.
4. **Serving Layer (`serve.py`)**: Sets up a FastAPI application that loads the `@champion` model and serves real-time prediction and health-check endpoints.

---

## Option 1: Container-Native Workflow (Recommended)

This option requires only Docker to be installed. The Dockerfile is designed to be fully self-contained: it runs data generation, model training, and model registration _during_ the build stage on a Linux base image, avoiding any OS-specific pathing conflicts.

### 1. Build the Docker Image

```bash
docker build -t loan-default-api .
```

### 2. Run the Container

```bash
docker run -p 8000:8000 loan-default-api
```

The server will boot up and automatically bind to `0.0.0.0:8000`, making it accessible from your host machine at `http://localhost:8000`.

---

## Option 2: Local Development Workflow (Without Docker)

If you want to run the pipeline step-by-step on your local host machine:

### 1. Environment Setup

Create and activate a Python virtual environment:

```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux Terminal
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the ML Pipeline

Execute the scripts in order to generate the dataset, train the models, and pin your champion model:

```bash
# Generate the synthetic dataset (creates data/loans.csv)
python generate_data.py

# Train models and log experiments to MLflow (creates mlflow.db & mlartifacts/)
python train.py

# Identify and register the champion model version
python register_model.py
```

### 4. Start the API Server

Start the local FastAPI application using Uvicorn:

```bash
uvicorn serve:app --reload
```

---

## Interacting with the API

Once your server is running (either locally or in Docker) at `http://127.0.0.1:8000`:

### 1. Open Interactive Documentation

Navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your web browser to access the interactive Swagger UI and test endpoints directly.

### 2. Check API Health

**Request:**

```bash
curl http://127.0.0.1:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "model": "models:/loan-default-classifier@champion"
}
```

### 3. Query a Prediction

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "annual_income": 55000,
    "loan_amount": 15000,
    "debt_to_income": 22.5,
    "employment_length": 3,
    "home_ownership": "RENT",
    "loan_purpose": "debt_consolidation",
    "interest_rate": 14.5,
    "open_accounts": 8,
    "delinquencies_2yr": 1
  }'
```

**Response:**

```json
{
  "default_probability": 0.5842,
  "prediction": "DEFAULT",
  "risk_level": "MEDIUM"
}
```

---

## CI/CD Pipeline

This repository is equipped with an automated GitHub Actions workflow (`.github/workflows/ci.yml`). On every push or pull request to the `main` branch, the pipeline automatically:

- Checks out the repository code.
- Builds the self-contained Docker image (handling database compilation, training, and registration in isolation).
- Launches the Docker container.
- Verifies both the `/health` and `/predict` REST endpoints using automated curl validation.
- Cleans up and stops running containers.
