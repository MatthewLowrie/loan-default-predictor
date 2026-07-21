import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# The tracking URI tells MLflow to store everything in a local SQLite database file rather than needing a separate server.
TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)


def load_data():
    """Load and split the loan dataset."""
    df = pd.read_csv("data/loans.csv")
    X = df.drop("loan_default", axis=1)
    y = df["loan_default"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify=y ensures the train/test split maintains the same default rate as the original dataset, which is important for imbalanced classification tasks.


def build_preprocessor():
    """
    
    Build a ColumnTransformer for mixed feature types.
    
    Your data has two types of features: numbers (credit score, income) and categories (home ownership, loan purpose). 
    ML models need them processed differently. 
    
    ColumnTransformer applies the right transformation to each column type in a single step.

    Numeric features get scaled so no single feature dominates by magnitude. 
    Categorical features get one-hot encoded into binary columns the model can understand.
    
    """
    
    numeric_features = [
        "credit_score",
        "annual_income",
        "loan_amount",
        "debt_to_income",
        "employment_length",
        "interest_rate",
        "open_accounts",
        "delinquencies_2yr",
    ]
    categorical_features = ["home_ownership", "loan_purpose"]

    # Scale numbers to same range, one-hot encode categories
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def train_model(model, model_name, X_train, X_test, y_train, y_test):
    
    """
    
    The Pipeline chains preprocessing and classification into one object. 
    When you call .fit(), it first transforms the data, then trains the classifier.

    The mlflow.start_run context manager creates a tracked experiment run. 
    Everything inside it (metrics, parameters, models) gets logged automatically to your MLflow database.
    
    """
    
    preprocessor = build_preprocessor()
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        # Generate predictions and probabilities
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        
        
   
    # This logs the model type, all metrics, and the full trained pipeline to MLflow. 
    # The infer_signature call records the input/output schema so MLflow knows what shape of data the model expects.

    # The print statements give you immediate terminal feedback showing each model's performance. 
    # The classification_report breaks down precision and recall for each class (Default vs No Default), which is where the real story shows up.
    
        mlflow.log_params({"model_type": model_name})
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(pipeline, "model", signature=signature)

        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return metrics


def main():
    
    """
    
    The baseline LogisticRegression treats both classes equally. 
    With 80% of records being non-defaults, it can get 80% accuracy by just predicting "no default" for everyone.

    The balanced models use class_weight="balanced", which tells the algorithm to penalize misclassifying defaults more heavily. 
    This forces the model to actually learn what defaults look like, even though they are the minority class.
    
    """ 
    mlflow.set_experiment("loan-default-prediction") # This creates an experiment and allows us to remember information about the run such as the model type, hyperparameters, and evaluation metrics. You can view all runs in the MLflow UI.

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Default rate (train): {y_train.mean():.1%}")
    print(f"Default rate (test): {y_test.mean():.1%}")

    # Model 1: Naive baseline with no class weight adjustment
    train_model(
        LogisticRegression(max_iter=1000),
        "logistic_regression_baseline",
        X_train, X_test, y_train, y_test,
    )

    # Model 2: Balanced weights to penalize missing defaults
    train_model(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        "logistic_regression_balanced",
        X_train, X_test, y_train, y_test,
    )

    # Model 3: Random forest with balanced weights
    train_model(
        RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "random_forest_balanced",
        X_train, X_test, y_train, y_test,
    )


if __name__ == "__main__":
    main()