import os
import yaml
import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
import numpy as np
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature


# Reproducibility

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Load Config

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"

print(f"Loading config from: {CONFIG_PATH}")

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# MLflow Setup

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])
mlflow.autolog(log_models=False)

# Load Data

data = pd.read_csv(config["data"]["path"])
target = config["data"]["target"]

assert target in data.columns, "Target column missing"
assert not data.isnull().any().any(), "Dataset contains null values"

X = data.drop(columns=[target])
y = data[target]


# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["train"]["test_size"],
    random_state=config["train"]["random_state"]
)

# Model

model = LinearRegression()

# Auto Model Versioning

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

existing_versions = [
    int(f.split("_v")[1].split(".pkl")[0])
    for f in os.listdir(MODEL_DIR)
    if f.startswith("model_v") and f.endswith(".pkl")
]

next_version = max(existing_versions) + 1 if existing_versions else 1
model_filename = f"model_v{next_version}.pkl"
model_path = os.path.join(MODEL_DIR, model_filename)


# Training

with mlflow.start_run(run_name=f"train_v{next_version}"):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # MLflow logging
    mlflow.log_param("model_version", next_version)
    mlflow.log_param("model_type", "LinearRegression")

    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)

    # Save model locally
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Update latest model
    latest_path = os.path.join(MODEL_DIR, "latest_model.pkl")
    with open(latest_path, "wb") as f:
        pickle.dump(model, f)

    # Log model to MLflow as artifact
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature
    )

    mlflow.set_tag("author", "Manish")
    mlflow.set_tag("use_case", "cricket_score_prediction")

print(f"Model trained & saved as {model_filename}")
