import os
import json
import argparse
import webbrowser
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.pyfunc import PyFuncModel

# ========================================
# 🔐 MLflow connection & environment setup
# ========================================
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "admin")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "password")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "https://mlflow.ml.brain.cs.ait.ac.th"))

MODEL_NAME = "st125999-a3-model"
DEFAULT_ALIAS = "Staging"

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, default=os.getenv("MODEL_VERSION"), help="Model version to load")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--port", type=int, default=8060)
args = parser.parse_args()

client = MlflowClient()

# ========================================
# 🔍 Resolve model URI
# ========================================
def resolve_model_uri():
    # 1️⃣ Explicit version (if provided)
    if args.version:
        print(f"🎯 Using explicit version: {args.version}")
        return f"models:/{MODEL_NAME}/{args.version}"

    # 2️⃣ Try alias (Staging)
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, DEFAULT_ALIAS)
        print(f"✅ Found alias '{DEFAULT_ALIAS}' -> version {mv.version}")
        return f"models:/{MODEL_NAME}@{DEFAULT_ALIAS}"
    except Exception:
        print(f"ℹ️ Alias '{DEFAULT_ALIAS}' not found. Trying newest READY version...")

    # 3️⃣ Fallback to newest READY version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    ready_versions = [v for v in versions if (v.status or '').upper() == 'READY'] or versions
    if not ready_versions:
        raise RuntimeError(f"❌ No versions found for model '{MODEL_NAME}'")

    latest = max(ready_versions, key=lambda v: int(v.version))
    print(f"✅ Using newest READY version: {latest.version}")
    return f"models:/{MODEL_NAME}/{latest.version}"

# ========================================
# 🔄 Load model and metadata
# ========================================
try:
    model_uri = resolve_model_uri()
    print(f"🔗 Loading model: {model_uri}")
    model_pipeline: PyFuncModel = mlflow.pyfunc.load_model(model_uri)
    print("✅ Model loaded successfully from MLflow!")

    # Load metadata assets (if any)
    run_id = model_pipeline.metadata.get_model_info().run_id
    local_assets_dir = client.download_artifacts(run_id, "", ".")
    assets_path = os.path.join(local_assets_dir, "assets.json")

    price_bin_edges = [0, 260000, 450000, 680000, 1000000]
    features = ["year", "max_power", "mileage", "brand", "fuel"]

    if os.path.exists(assets_path):
        with open(assets_path, "r") as f:
            assets = json.load(f)
        price_bin_edges = assets.get("price_bin_edges", price_bin_edges)
        num_cols = assets.get("num_cols")
        cat_cols = assets.get("cat_cols")
        if num_cols and cat_cols:
            features = num_cols + cat_cols

    print("✅ Loaded feature metadata:", features)

except Exception as e:
    print(f"🚨 Error loading model or assets: {e}")
    model_pipeline = None
    price_bin_edges = [0, 260000, 450000, 680000, 1000000]
    features = ["year", "max_power", "mileage", "brand", "fuel"]

# ========================================
# 🎨 Dash UI
# ========================================
app = dash.Dash(__name__)
app.title = "Car Price Class Predictor"

if model_pipeline is None:
    app.layout = html.Div([
        html.H1("🚨 Could not load model", style={"color": "red", "textAlign": "center"}),
        html.P(f"Model: {MODEL_NAME}. Try passing --version N or set MODEL_VERSION in env.",
               style={"textAlign": "center"}),
    ])
else:
    brand_options = [{'label': b, 'value': b} for b in [
        'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota',
        'Chevrolet', 'Renault', 'Volkswagen', 'Nissan', 'Skoda', 'BMW', 'Mercedes-Benz'
    ]]
    fuel_options = [{'label': f, 'value': f} for f in ['Petrol', 'Diesel', 'Electric']]

    app.layout = html.Div([
        html.Div([
            html.H1("🚗 Car Price Class Predictor", style={"margin": "0"}),
            html.P("Predict the price class (0–3) of a used car.", style={"margin": "0"})
        ], style={"backgroundColor": "#007BFF", "color": "white", "padding": "20px", "textAlign": "center"}),

        html.Div([
            html.Div([
                html.H3("Enter Car Details", style={"marginBottom": "20px"}),

                html.Label("Year of Manufacture:"),
                dcc.Input(id="input-year", type="number", placeholder="e.g., 2018",
                          style={"width": "100%", "marginBottom": "10px"}),

                html.Label("Max Power (bhp):"),
                dcc.Input(id="input-max_power", type="number", placeholder="e.g., 85",
                          style={"width": "100%", "marginBottom": "10px"}),

                html.Label("Mileage (kmpl):"),
                dcc.Input(id="input-mileage", type="number", placeholder="e.g., 17.5",
                          style={"width": "100%", "marginBottom": "10px"}),

                html.Label("Brand:"),
                dcc.Dropdown(id="input-brand", options=brand_options, value="Maruti",
                             clearable=False, style={"marginBottom": "10px"}),

                html.Label("Fuel Type:"),
                dcc.Dropdown(id="input-fuel", options=fuel_options, value="Petrol",
                             clearable=False, style={"marginBottom": "20px"}),

                html.Button("🔮 Predict Class", id="predict-btn", n_clicks=0,
                            style={"backgroundColor": "#28a745", "color": "white",
                                   "padding": "10px 20px", "border": "none", "cursor": "pointer"}),

                html.Div(id="prediction-output",
                         style={"marginTop": "30px", "fontSize": "20px", "fontWeight": "bold"})
            ], style={"width": "400px", "padding": "30px", "backgroundColor": "#f8f9fa",
                      "borderRadius": "10px", "boxShadow": "0 0 10px rgba(0,0,0,0.1)"})
        ], style={"display": "flex", "justifyContent": "center", "marginTop": "50px"})
    ])

    # ========================================
    # 🧠 Callback: Prediction Logic
    # ========================================
    @app.callback(
        Output("prediction-output", "children"),
        Input("predict-btn", "n_clicks"),
        [
            State("input-year", "value"),
            State("input-max_power", "value"),
            State("input-mileage", "value"),
            State("input-brand", "value"),
            State("input-fuel", "value"),
        ]
    )
    def predict_price(n_clicks, year, max_power, mileage, brand, fuel):
        if n_clicks == 0:
            return ""
        if None in [year, max_power, mileage, brand, fuel]:
            return html.Span("⚠️ Please fill all fields.", style={"color": "red"})

        try:
            # 1️⃣ Build DataFrame
            payload = {"year": int(year), "max_power": float(max_power),
                       "mileage": float(mileage), "brand": brand, "fuel": fuel}
            df_input = pd.DataFrame([payload])[features]

            # 2️⃣ Encode categoricals
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            le_brand = LabelEncoder()
            le_fuel = LabelEncoder()
            le_brand.fit([
                'Maruti', 'Hyundai', 'Mahindra', 'Tata', 'Honda', 'Ford', 'Toyota',
                'Chevrolet', 'Renault', 'Volkswagen', 'Nissan', 'Skoda', 'BMW', 'Mercedes-Benz'
            ])
            le_fuel.fit(['Petrol', 'Diesel', 'Electric'])
            df_input["brand"] = le_brand.transform(df_input["brand"])
            df_input["fuel"] = le_fuel.transform(df_input["fuel"])

            # 3️⃣ Scale + Add intercept
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_input.values)
            X_infer = np.c_[np.ones((X_scaled.shape[0], 1)), X_scaled]

            # 4️⃣ Predict
            pred_class = int(model_pipeline.predict(X_infer)[0])

            # ✅ Show raw class number
            category = str(pred_class)

            lower = price_bin_edges[pred_class]
            upper = price_bin_edges[pred_class + 1] if pred_class + 1 < len(price_bin_edges) else price_bin_edges[-1]

            return html.Div([
                html.Span("Predicted Class: ", style={"color": "#333"}),
                html.Span(category, style={"color": "#007BFF"}),

                html.Br(),
                html.Span("Estimated Price Range: ", style={"color": "#333"}),
                html.Span(f"₹{lower:,.0f} - ₹{upper:,.0f}", style={"color": "#28a745"}),
            ])

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return html.Span(f"❌ Error: {str(e)}", style={"color": "red"})

# ========================================
# 🚀 Run Server (Dash 2.16+)
# ========================================
if __name__ == "__main__":
    url = f"http://{args.host}:{args.port}"
    print(f"🚀 Starting Dash at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    app.run(host=args.host, port=args.port, debug=True)
