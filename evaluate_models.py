import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd
import numpy as np
import os
import joblib

# change what model you want to use (registered model name)
MODEL_NAME = "IRIS-Classifier-LogReg"

# Use the environment variable set by Kubernetes, with a local fallback
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

def fetch_and_load_best_model(model_name: str):
    """
    Fetches the model version currently marked as 'Production' in MLflow Registry and loads it.
    
    This is the standard and most reliable method for deployment.
    """
    try:
        # Request only models currently marked as 'Production'
        versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not versions:
            # This is the error point we keep hitting. It means MLflow cannot find the Production stage.
            raise ValueError(
                f"No model versions found in 'Production' stage for registered model: {model_name}. "
                "Ensure the model is explicitly transitioned to 'Production' in the MLflow UI."
            )
        
        # Load the latest model version found in the Production stage
        latest_version = versions[0]
        model_uri = latest_version.source
        
        print(f"Loading model '{model_name}' version {latest_version.version}, stage: {latest_version.current_stage}")
        
        # Load the model directly from the artifact URI
        model = mlflow.sklearn.load_model(model_uri)
        return model

    except MlflowException as e:
        print(f"MLflow error while fetching model '{model_name}': {e}")
        raise
    except Exception as e:
        print(f"Unexpected error during model loading: {e}")
        raise

# Placeholder function for compatibility with app.py's other calls
def fetch_and_load_latest_model(model_name: str):
    # For deployment, 'best' is defined as 'Production'
    return fetch_and_load_best_model(model_name)

def run_evaluation(model_type:str):
    # This is only a placeholder function to satisfy imports, it does nothing in the deployment API
    pass