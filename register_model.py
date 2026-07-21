import mlflow
from mlflow import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)

def register_best_model():
    """
    
    Find the best model and register it with the champion alias.
    
    MlflowClient() creates a client that can query the tracking database and manage the model registry.
    get_experiment_by_name looks up the experiment you created in train.py.
    search_runs queries all runs, sorted by ROC-AUC descending, and returns only the top one.
    
    
    best_run.data.params and best_run.data.metrics pull the parameters and metrics you logged during training.
    mlflow.register_model copies the model artifact into the registry under the name loan-default-classifier.
    set_registered_model_alias points the champion alias at this version. Any code loading models:/loan-default-classifier@champion now gets this exact model.
   
    """
    
    
    client = MlflowClient()
    experiment = client.get_experiment_by_name("loan-default-prediction")
    
    if experiment is None:
        print("No experiment found. Please run the training script first.")
        return

    runs = client.search_runs(experiment.experiment_id, order_by=["metrics.rou_auc DESC"],max_results=1,
                              )
    
    if not runs:
        print("No runs found. Please run the training script first.")
        return
    
    
    
    best_run = runs[0]
    model_type = best_run.data.params.get("model_type", "unknown")
    roc_auc = best_run.data.metrics["roc_auc"]
    print(f"Best model: {model_type}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "loan-default-classifier"

    result = mlflow.register_model(model_uri, model_name)
    print(f"Registered model version: {result.version}")

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=result.version,
    )
    print(f"Set 'champion' alias to version {result.version}")


if __name__ == "__main__":
    register_best_model()