import yaml
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import ModelFactory

def robust_load_config(path: str = "config/config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback default config if file is missing (e.g. in tests)
        return {
            "SYMBOLS": ["EURUSD"],
            "TIMEFRAME": "1h",
            "LIGHTGBM_PARAMS": {"objective": "multiclass", "num_class": 3, "verbose": -1},
            "CNN_PARAMS": {"sequence_length": 60, "epochs": 5, "batch_size": 32}
        }

def train_pipeline():
    # Load Config
    config = robust_load_config()
    
    # Initialize
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    trainer = ModelFactory(config)
    
    symbols = config.get("SYMBOLS", ["EURUSD"])
    interval = config.get("TIMEFRAME", "60min")
    seq_len = config.get("CNN_PARAMS", {}).get("sequence_length", 60)
    
    # 1. Collect and Process Data
    processed_dfs = []
    print("Fetching and processing data...")
    
    for symbol in symbols:
        df = loader.fetch_data(symbol, interval=interval)
        if df.empty:
            print(f"Skipping {symbol} (no data)")
            continue
            
        # Add indicators
        df = preprocessor.add_technical_indicators(df)
        
        # Create targets
        df = preprocessor.create_labels(df)
        
        if df.empty:
             print(f"Skipping {symbol} (empty after processing)")
             continue
             
        processed_dfs.append(df)
            
    if not processed_dfs:
        print("No valid data found for any symbol.")
        return

    # 2. Fit Scaler locally on ALL data stacked
    print("Fitting scaler...")
    all_features = []
    # Assume all DFs have same columns
    feature_cols = [c for c in processed_dfs[0].columns if c not in ['target', 'date']]
    
    for df in processed_dfs:
        all_features.append(df[feature_cols].values)
        
    big_X = np.concatenate(all_features)
    preprocessor.scaler.fit(big_X)
    print(f"Scaler fitted on {len(big_X)} samples.")
    
    # 3. Transform and Prepare Datasets
    all_X_lgbm = []
    all_y_lgbm = []
    all_X_cnn = []
    all_y_cnn = []
    
    for df in processed_dfs:
        X = df[feature_cols].values
        y = df['target'].values
        
        # Transform
        X_scaled = preprocessor.scaler.transform(X)
        
        # LGBM Data (Simple rows)
        all_X_lgbm.append(X_scaled)
        all_y_lgbm.append(y)
        
        # CNN Data (Sequences)
        # We need enough data for at least one sequence
        if len(X_scaled) > seq_len:
            Xs, ys = preprocessor.create_sequences(X_scaled, y, lookback=seq_len)
            all_X_cnn.append(Xs)
            all_y_cnn.append(ys)
        
    if not all_X_lgbm:
        print("Not enough data for training.")
        return

    X_lgbm = np.concatenate(all_X_lgbm)
    y_lgbm = np.concatenate(all_y_lgbm)
    
    if all_X_cnn:
        X_cnn = np.concatenate(all_X_cnn)
        y_cnn = np.concatenate(all_y_cnn)
    else:
        print("Warning: No data for CNN (sequences too short?)")
        X_cnn = np.array([])
        y_cnn = np.array([])
    
    # Split
    X_train_l, X_val_l, y_train_l, y_val_l = train_test_split(X_lgbm, y_lgbm, test_size=0.2, shuffle=False)
    
    # Train LightGBM
    print("Training LightGBM...")
    lgbm_model = trainer.train_lightgbm(X_train_l, y_train_l, X_val_l, y_val_l)
    trainer.save_lgbm(lgbm_model, "models/lgbm_model.txt")
    print("LightGBM saved.")
    
    # Train CNN (Only if TF is available)
    from src.models import HAS_TF
    if HAS_TF and len(X_cnn) > 0:
        X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_cnn, y_cnn, test_size=0.2, shuffle=False)
        
        print("Training CNN...")
        input_shape = (X_train_c.shape[1], X_train_c.shape[2])
        try:
            cnn_model = trainer.build_cnn(input_shape)
            cnn_callbacks = trainer.get_callbacks()
            
            cnn_params = config.get("CNN_PARAMS", {})
            cnn_model.fit(
                X_train_c, y_train_c,
                validation_data=(X_val_c, y_val_c),
                epochs=cnn_params.get("epochs", 5),
                batch_size=cnn_params.get("batch_size", 32),
                callbacks=cnn_callbacks
            )
            trainer.save_cnn(cnn_model, "models/cnn_model.keras")
            print("CNN saved.")
        except Exception as e:
            print(f"CNN training failed: {e}")
    else:
        print("Skipping CNN training (TensorFlow not found or no data).")

    
    # Save Scaler for Inference
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(preprocessor.scaler, f)
    print("Scaler saved.")

if __name__ == "__main__":
    train_pipeline()
