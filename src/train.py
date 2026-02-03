import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("cricket_score_prediction")


experiment = mlflow.get_experiment_by_name("cricket_score_prediction")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
run_name = f"Cricket Score v{len(runs) + 1}"


data = pd.read_csv("data/cricket_data.csv")

X = data.drop("final_score", axis=1)
y = data["final_score"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


with mlflow.start_run(run_name=run_name) as run:

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log metrics and params
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Infer signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model + register
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="cricket_score_model",
        signature=signature
    )

print("Model trained and registered successfully")
