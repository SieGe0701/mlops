
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from mlflow.models.signature import infer_signature

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Prepare input_example for logging
input_example = X_test[:2]

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)

    # Infer model signature
    signature = infer_signature(X_test, preds)

    # Log model with input_example and signature, using 'name' instead of deprecated 'artifact_path'
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        signature=signature
    )
    print(f"Logged run with accuracy: {acc}")
