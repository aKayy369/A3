import mlflow
from mlflow.tracking import MlflowClient
import os

# ======================================================
# 🔐 Authenticate and Connect to MLflow
# ======================================================
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "password")

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# ======================================================
# 🧱 Model Info
# ======================================================
MODEL_NAME = "st125999-a3-model"
STAGING_STAGE = "Staging"
PRODUCTION_STAGE = "Production"

# ======================================================
# 🔍 Step 1: Find the latest model in Staging
# ======================================================
staging_versions = client.get_latest_versions(MODEL_NAME, [STAGING_STAGE])

if not staging_versions:
    print("❌ No model currently in 'Staging'. Promote one first.")
    exit(1)

staging_model = staging_versions[0]
version_number = staging_model.version
run_id = staging_model.run_id

print(f"🔍 Found model in Staging: version={version_number}, run_id={run_id}")

# ======================================================
# 🚀 Step 2: Transition it to Production
# ======================================================
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=version_number,
    stage=PRODUCTION_STAGE,
    archive_existing_versions=True
)

print(f"✅ Successfully promoted version {version_number} to 'Production'!")
