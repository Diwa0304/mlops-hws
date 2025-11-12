from evaluate_models import run_evaluation, fetch_and_load_latest_model, fetch_and_load_best_model
import joblib
import pandas as pd
import numpy as np
import os
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from typing import List
import uvicorn
import logging
import time
import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# setup tracker
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Setup structured logging
logger = logging.getLogger("demo-log-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()

formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- CRITICAL MLFLOW CONFIGURATION ---
# This MLFLOW_TRACKING_URI is now correctly passed via the deployment.yaml environment variable
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# -------------------------------------

MODEL_NAME = "IRIS-Classifier-LogReg"
ENCODER_PATH = "models/encoder.joblib" 

FEATURE_NAMES = [ "sepal_length", "sepal_width", "petal_length", "petal_width" ]

app = FastAPI()

model = None
encoder = None

def load_model_from_file_artifact(uri):
    """
    Loads the model directly from the artifact URI, attempting both 'runs:/' and 'mlflow-artifacts:/' formats.
    """
    
    # Define URIs used for robust loading attempts
    # Replace '1b5b32220242428d9a6d737c3bc97dee' with your actual Run ID if it changes, 
    # otherwise, use this stable one.
    RUN_ID = "1b5b32220242428d9a6d737c3bc97dee"
    EXPERIMENT_ID = "827514430569397916"
    
    model_uri_runs = f"runs:/{RUN_ID}/artifacts"
    model_uri_full_artifacts = f"mlflow-artifacts:/{EXPERIMENT_ID}/{RUN_ID}/artifacts"

    try:
        # 1. Try simplified 'runs:/' notation (Standard MLflow way)
        loaded_model = mlflow.sklearn.load_model(model_uri_runs)
        print(f"Model loaded successfully from URI (Runs Attempt): {model_uri_runs}")
        return loaded_model
    except MlflowException as e:
        # If 'runs:/' fails (due to database issue), try the full artifact path
        print(f"Runs URI failed. Attempting full artifact path: {model_uri_full_artifacts}")
        
        try:
            # 2. Try full 'mlflow-artifacts:/' format (Bypasses MLflow DB, relies on GCS/S3 direct download)
            loaded_model = mlflow.sklearn.load_model(model_uri_full_artifacts)
            print(f"Model loaded successfully from URI (Full Artifacts Path): {model_uri_full_artifacts}")
            return loaded_model
        except Exception as e_full:
            print(f"Error loading model from URI {model_uri_full_artifacts}: {e_full}")
            raise RuntimeError(f"Deployment failed: Cannot load model directly from Full Artifact Path. Error: {e_full}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Deployment failed: General error during model loading. Error: {e}")

try:
    # Pass a dummy value to the function, as the URIs are now hardcoded internally
    model = load_model_from_file_artifact("")
    encoder = joblib.load(ENCODER_PATH)

    print("API Model and Encoder loaded successfully.")
except FileNotFoundError:
    print(f"Error: Encoder file not found at {ENCODER_PATH}. Ensure train.py was run and the file is in the Docker image.")
    raise
except RuntimeError as e:
    # Catches the error from load_model_from_file_artifact
    print(f"Deployment failed: {e}")
    raise
except Exception as e:
    print(f"Fatal error during startup: {e}")
    raise
    
class FeatureInput(BaseModel):
    features: List[List[float]] 

app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    import time
    time.sleep(2)
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.get('/health')
def health_check():
    """Simple endpoint to check if the API is running and ready."""
    return {"status": "ok", "model_loaded": (model is not None)}, 200

@app.post('/predict')
async def predict(input_data: FeatureInput):
    """
    Handles POST requests with JSON data for prediction.
    """
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            df = pd.DataFrame(input_data.features, columns=FEATURE_NAMES)
            prediction_int = model.predict(df)
            prediction_species = encoder.inverse_transform(prediction_int.reshape(-1, 1)).flatten().tolist()
            latency = round((time.time() - start_time) * 1000, 2)
    
            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": input_data.features,
                "result": prediction_species,
                "latency_ms": latency,
                "status": "success"
            }))
            return {
                    "status": "success",
                    "predictions": prediction_species
                }
        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0', port=8000)