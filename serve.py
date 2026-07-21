import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# mlflow.set_tracking_uri(TRACKING_URI) tells MLflow where to find the model registry (the same mlflow.db from training).

# models:/loan-default-classifier@champion resolves the champion alias to whichever model version it points to. If you promote a new model later, the API picks it up automatically on restart.

# The model loads once at the module level, not per request. Loading from MLflow takes seconds, so doing it per-request would make every prediction unusably slow.




# Point MLflow to the same tracking database used during training 
TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)

# Create the FastAPI application
app = FastAPI(
    title="Loan Default Prediction API",
    description="Predict whether an applicant is likely to default on a loan based on their financial profile.",
    version="1.0.0",
)


# Load the champion model once at startup,  
# Loading from MLflow takes seconds, so doing it per-request would make every prediction unusably slow.
MODEL_NAME = "loan-default-classifier"
MODEL_URI = f"models:/{MODEL_NAME}@champion"
model = mlflow.pyfunc.load_model(MODEL_URI)

# Define the expected shape of a loan application request
class LoanApplication(BaseModel):
    credit_score: int
    annual_income: float
    loan_amount: float
    debt_to_income: float
    employment_length: int
    home_ownership: str
    loan_purpose: str
    interest_rate: float
    open_accounts: int
    delinquencies_2yr: int


# Define the shape of the prediction response
class PredictionResponse(BaseModel):
    default_probability: float
    prediction: str
    risk_level: str
    

# /health is a simple GET endpoint that confirms the API is running and which model it loaded. 
# This is the endpoint you'll use later for automated health checks.
@app.get("/health")
def health_check():
    return {"status": "healthy", "model": MODEL_URI}

# /predict accepts a POST request with a JSON loan application. 
# It converts the input to a DataFrame (what the sklearn pipeline expects), 
# runs the model, then maps the raw probability to a human-readable prediction and risk level.
@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    try:
        # Convert request to DataFrame and get prediction probability
        input_df = pd.DataFrame([application.model_dump()])
        proba = model.predict(input_df)
        probability = float(proba[0])

    # The 0.5 threshold for DEFAULT/NO DEFAULT and the 0.3/0.6 bands for risk levels are business decisions. 
    # In production, these thresholds would be tuned based on the bank's risk appetite.




        # Map probability to a binary prediction
        if probability >= 0.5:
            prediction = "DEFAULT"
        else:
            prediction = "NO DEFAULT"

        # Assign risk level based on probability bands
        if probability < 0.3:
            risk_level = "LOW"
        elif probability < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            default_probability=round(probability, 4),
            prediction=prediction,
            risk_level=risk_level,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
    # uvicorn serve:app --reload