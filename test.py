import unittest
import pandas as pd
import numpy as np
import os
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ======================================================
# 🔐 MLflow Configuration
# ======================================================
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "password")

MODEL_NAME = "st125999-a3-model"
MODEL_ALIAS = "Staging"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


# ======================================================
# 🧪 Unit Test Class
# ======================================================
class TestDeployedModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load model once before all tests"""
        print(f"\n--- Loading model '{MODEL_NAME}' (alias '{MODEL_ALIAS}') ---")
        cls.model = mlflow.pyfunc.load_model(MODEL_URI)
        print("✅ Model loaded successfully.")

        cls.features = ["year", "max_power", "mileage", "brand", "fuel"]

        # prepare encoders and scaler like inference
        cls.le_brand = LabelEncoder()
        cls.le_fuel = LabelEncoder()
        cls.scaler = StandardScaler()

    # 🔁 helper function to preprocess
    def preprocess(self, df):
        df = df.copy()
        df["brand"] = self.le_brand.fit_transform(df["brand"])
        df["fuel"] = self.le_fuel.fit_transform(df["fuel"])
        X_scaled = self.scaler.fit_transform(df)
        X_final = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]  # add bias
        return X_final

    def test_model_takes_expected_input(self):
        """✅ Test 1: check model runs with 1 valid input"""
        print("\nRunning Test 1: Model takes expected input...")

        df = pd.DataFrame([{
            "year": 2019,
            "max_power": 95.0,
            "mileage": 21.5,
            "brand": "Maruti",
            "fuel": "Petrol"
        }])

        X_infer = self.preprocess(df)
        preds = self.model.predict(X_infer)

        self.assertIsNotNone(preds)
        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (1,))
        print("  -> ✅ PASSED")

    def test_output_has_expected_shape(self):
        """✅ Test 2: check model runs with 3-row batch"""
        print("\nRunning Test 2: Output has expected shape...")

        df = pd.DataFrame([
            {"year": 2015, "max_power": 70.0, "mileage": 18.0, "brand": "Tata", "fuel": "Diesel"},
            {"year": 2018, "max_power": 90.0, "mileage": 22.0, "brand": "Honda", "fuel": "Petrol"},
            {"year": 2022, "max_power": 150.0, "mileage": 15.0, "brand": "Toyota", "fuel": "Petrol"}
        ])

        X_infer = self.preprocess(df)
        preds = self.model.predict(X_infer)

        self.assertIsInstance(preds, np.ndarray)
        self.assertEqual(preds.shape, (3,))
        print("  -> ✅ PASSED")


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
