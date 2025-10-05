import os
import mlflow
from mlflow.tracking import MlflowClient

# 🔐 MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

# 🌐 Connect to your MLflow server
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th")

MODEL_NAME = "st125999-a3-model"
VERSION = 8  # 👈 use version 8

client = MlflowClient()

print(f"🚀 Promoting {MODEL_NAME} version {VERSION} → STAGING...")

# 🏷️ Promote version 8 to Staging (and archive old ones)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=VERSION,
    stage="Staging",
    archive_existing_versions=True
)

print(f"✅ Version {VERSION} promoted to STAGING stage successfully!")
