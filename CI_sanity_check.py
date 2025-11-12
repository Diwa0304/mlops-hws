import pandas as pd
import mlflow.pyfunc
from sklearn.metrics import accuracy_score
from mlflow import MlflowClient

def fetch_and_load_best_model(model_name: str, metric_key: str = "accuracy"):
    """
    Fetches the model version with the highest value for a specific metric.
    """
    # mlflow.set_tracking_uri("http://127.0.0.1:8100")
    client = MlflowClient()
    
    all_versions = client.search_model_versions(filter_string=f"name='{model_name}'")
    if not all_versions:
        raise ValueError(f"No model versions found for registered model: {model_name}")
        
    best_acc = -1.0
    best_model_version = None
        
    for mv in all_versions:
        try:
            run = client.get_run(mv.run_id)
            current_acc = run.data.metrics.get(metric_key, -2.0)
            
            if current_acc > best_acc:
                best_acc = current_acc
                best_model_version = mv.version
                
        except MlflowException as e:
            print(f"Warning: Could not fetch run metrics for version {mv.version}: {e}")
            continue

    if best_model_version is None or best_acc == -1.0:
        raise MlflowException(f"Failed to find a suitable model version with metric '{metric_key}'.")

    print(f"Best model found: Version {best_model_version} with {metric_key}={best_acc:.4f}")

    model_uri = f"models:/{model_name}/{best_model_version}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    return loaded_model

def run_sanity_check():
    model_name = "IRIS-Classifier-LogReg"
    
    model = fetch_and_load_best_model(model_name)
    
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    MIN_ACCURACY_THRESHOLD = 0.90
    assert acc >= MIN_ACCURACY_THRESHOLD, f"CI Sanity Check Failed: Best registered model accuracy ({acc:.4f}) is below the required minimum ({MIN_ACCURACY_THRESHOLD})."
    
    print(f"CI Sanity Check Passed! Best registered model accuracy: {acc:.4f}")
    
    
if __name__ == "__main__":
    run_sanity_check()
