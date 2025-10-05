import os
import mlflow
from mlflow.tracking import MlflowClient

# ======================================================
# 🔐 MLflow Configuration
# ======================================================
print("🔗 Connecting to MLflow...")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "password")

if not MLFLOW_TRACKING_USERNAME or not MLFLOW_TRACKING_PASSWORD:
    raise ValueError("❌ Missing MLflow credentials (username/password).")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD

MODEL_NAME = "st125999-a3-model"
STAGING_ALIAS = "Staging"
PRODUCTION_ALIAS = "Production"

client = MlflowClient()

def promote_staging_to_production():
    try:
        # ✅ Step 1: Get model version with alias 'Staging'
        print(f"🔍 Searching for '{STAGING_ALIAS}' alias in model '{MODEL_NAME}'...")
        model_version = client.get_model_version_by_alias(MODEL_NAME, STAGING_ALIAS)
        version = model_version.version
        print(f"✅ Found Version {version} currently in Staging.")

    except Exception:
        # 🚨 If no Staging alias exists, pick the latest version
        print("⚠️ No model found in Staging. Using latest version instead.")
        latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not latest_versions:
            raise Exception("❌ No model versions found in registry.")
        version = max(int(v.version) for v in latest_versions)
        print(f"✅ Using latest version: {version}")
        # Assign alias Staging first (optional)
        client.set_registered_model_alias(MODEL_NAME, STAGING_ALIAS, version)

    # ✅ Step 2: Promote to Production
    print(f"🚀 Promoting Version {version} to '{PRODUCTION_ALIAS}'...")
    client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
    print(f"🎉 Model '{MODEL_NAME}' Version {version} promoted to '{PRODUCTION_ALIAS}' successfully!")

if __name__ == "__main__":
    promote_staging_to_production()
