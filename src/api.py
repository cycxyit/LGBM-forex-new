from fastapi import FastAPI, HTTPException
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import yaml
import os

# TensorFlow might be missing in some envs
try:
    from tensorflow.keras.models import load_model # types: ignore
    HAS_TF = True
except ImportError:
    HAS_TF = False

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor

app = FastAPI(title="AI Forex Trading API")

# Global Load
config_path = "config/config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize modules
loader = DataLoader(config_path)
preprocessor = DataPreprocessor()

# Load Inteference Artifacts
models = {}
scaler = None

@app.on_event("startup")
def load_artifacts():
    global models, scaler
    
    # Load Scaler
    if os.path.exists("models/scaler.pkl"):
        try:
            with open("models/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
                preprocessor.scaler = scaler # Set pre-fitted scaler
        except Exception as e:
            print(f"Error loading scaler: {e}")
    else:
        print("Warning: Scaler not found. Run training first.")

    # Load LightGBM
    if os.path.exists("models/lgbm_model.txt"):
        models['lgbm'] = lgb.Booster(model_file="models/lgbm_model.txt")
    
    # Load CNN
    if HAS_TF and os.path.exists("models/cnn_model.keras"):
        models['cnn'] = load_model("models/cnn_model.keras")

@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.get("/predict/{symbol}")
def predict(symbol: str):
    """
    Get prediction for a symbol.
    Fetches latest data, processes it, and runs inference.
    """
    if not scaler:
         raise HTTPException(status_code=503, detail="Models not loaded (Scaler missing)")
         
    # Fetch data (need enough for lookback and indicators)
    # Fetch data (need enough for lookback and indicators)
    # 60 (lookback) + 50 (indicators warm up) ~= 150 points
    # yfinance default period is 2y, which is plenty.
    df = loader.fetch_data(symbol, interval=config.get("TIMEFRAME", "60min"), period="6mo")
    
    if df.empty:
        raise HTTPException(status_code=404, detail="Symbol data not found")
        
    # Preprocess
    df = preprocessor.add_technical_indicators(df)
    
    if len(df) < 60: # Minimum required for sequence
        raise HTTPException(status_code=400, detail="Not enough data points after indicator calculation")
        
    # Prepare features
    feature_cols = [c for c in df.columns if c not in ['target', 'date', 'close_target']] # exclude potential targets
    
    # We take the *latest* sequence/row for prediction
    # Actually, we need a sequence for CNN.
    
    # Scale ALL data to match fit distribution? 
    # Yes, using the pre-fitted scaler.
    
    # Extract last row features for LightGBM
    last_row = df.iloc[[-1]][feature_cols]
    X_lgbm = scaler.transform(last_row.values)
    
    # Extract last 60 rows for CNN
    cnn_seq = df.iloc[-60:][feature_cols]
    if len(cnn_seq) < 60:
         raise HTTPException(status_code=400, detail="Insufficient data for CNN sequence")
         
    X_cnn_raw = scaler.transform(cnn_seq.values)
    X_cnn = np.expand_dims(X_cnn_raw, axis=0) # [1, 60, features]
    
    response = {"symbol": symbol, "predictions": {}}
    
    # LightGBM Prediction
    if 'lgbm' in models:
        lgb_prob = models['lgbm'].predict(X_lgbm)[0]
        response["predictions"]["lightgbm"] = {
            "bearish_prob": float(lgb_prob[0]),
            "neutral_prob": float(lgb_prob[1]),
            "bullish_prob": float(lgb_prob[2]),
            "signal": ["Bearish", "Neutral", "Bullish"][np.argmax(lgb_prob)]
        }
        
    # CNN Prediction
    if 'cnn' in models:
        cnn_prob = models['cnn'].predict(X_cnn)[0]
        response["predictions"]["cnn"] = {
            "bearish_prob": float(cnn_prob[0]),
            "neutral_prob": float(cnn_prob[1]),
            "bullish_prob": float(cnn_prob[2]),
            "signal": ["Bearish", "Neutral", "Bullish"][np.argmax(cnn_prob)]
        }
        
    # Ensemble / Meta-Signal (Simple Average)
    if 'lgbm' in models and 'cnn' in models:
        avg_prob = (lgb_prob + cnn_prob) / 2
        response["predictions"]["ensemble"] = {
             "bearish_prob": float(avg_prob[0]),
             "bullish_prob": float(avg_prob[2]),
             "signal": ["Bearish", "Neutral", "Bullish"][np.argmax(avg_prob)]
        }
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
