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
    batch_size = config.get("CNN_PARAMS", {}).get("batch_size", 32)
    cnn_epochs = config.get("CNN_PARAMS", {}).get("epochs", 5)
    
    # 1. Collect Data (Keep as list of DF for now - metadata overhead is low)
    # If even this is too big (e.g. 50 years of data), we'd need to stream from disk.
    # For 5 years @ 1h/1m, holding raw DFs in RAM is usually fine (hundreds of MBs).
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

    # 2. Fit Scaler Incrementally (partial_fit) on chunks
    print("Fitting scaler incrementally...")
    # Assume all DFs have same columns
    feature_cols = [c for c in processed_dfs[0].columns if c not in ['target', 'date']]
    
    for df in processed_dfs:
        X = df[feature_cols].values
        # Partial fit expects 2D array
        preprocessor.scaler.partial_fit(X)
        
    print("Scaler fitted.")
    
    # Save Scaler early
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(preprocessor.scaler, f)
    print("Scaler saved.")

    # 3. Transform and Prepare Datasets (LGBM in-memory, CNN via Generators)
    full_X_lgbm_train = []
    full_y_lgbm_train = []
    full_X_lgbm_val = []
    full_y_lgbm_val = []
    
    train_datasets = []
    val_datasets = []
    
    # Check for TensorFlow availability
    from src.models import HAS_TF
    if HAS_TF:
        import tensorflow as tf
    else:
        print("TensorFlow not found. CNN will be skipped.")
    
    for df in processed_dfs:
        X = df[feature_cols].values
        y = df['target'].values
        
        # Transform (Now safe to transform since scaler is fitted)
        X_scaled = preprocessor.scaler.transform(X)
        
        # Split into Train/Val by time (e.g. 80/20)
        split_idx = int(len(X_scaled) * 0.8)
        
        X_train = X_scaled[:split_idx]
        y_train = y[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y[split_idx:]
        
        # --- LGBM Data Collection ---
        full_X_lgbm_train.append(X_train)
        full_y_lgbm_train.append(y_train)
        full_X_lgbm_val.append(X_val)
        full_y_lgbm_val.append(y_val)
        
        # --- CNN Dataset Creation ---
        if HAS_TF and len(X_train) > seq_len:
             # Train Dataset
             ds_train = preprocessor.create_tf_dataset(
                 X_train, y_train, 
                 lookback=seq_len, 
                 batch_size=batch_size, 
                 shuffle=True
             )
             train_datasets.append(ds_train)
             
             # Val Dataset
             if len(X_val) > seq_len:
                 ds_val = preprocessor.create_tf_dataset(
                     X_val, y_val, 
                     lookback=seq_len, 
                     batch_size=batch_size, 
                     shuffle=False
                 )
                 val_datasets.append(ds_val)

    # 4. Train LightGBM
    if full_X_lgbm_train:
        print("Training LightGBM...")
        X_train_l = np.concatenate(full_X_lgbm_train)
        y_train_l = np.concatenate(full_y_lgbm_train)
        X_val_l = np.concatenate(full_X_lgbm_val)
        y_val_l = np.concatenate(full_y_lgbm_val)
        
        lgbm_model = trainer.train_lightgbm(X_train_l, y_train_l, X_val_l, y_val_l)
        trainer.save_lgbm(lgbm_model, "models/lgbm_model.txt")
        print("LightGBM saved.")
        
        # Free memory
        del X_train_l, y_train_l, X_val_l, y_val_l
        import gc; gc.collect()
    else:
        print("No data for LightGBM.")

    # 5. Train CNN
    if HAS_TF and train_datasets:
        print("Training CNN...")
        # Combine datasets
        final_train_ds = train_datasets[0]
        for ds in train_datasets[1:]:
            final_train_ds = final_train_ds.concatenate(ds)
            
        # Shuffle global dataset roughly
        final_train_ds = final_train_ds.shuffle(buffer_size=1000)
            
        final_val_ds = None
        if val_datasets:
            final_val_ds = val_datasets[0]
            for ds in val_datasets[1:]:
                final_val_ds = final_val_ds.concatenate(ds)
        
        # Build Model
        # Input shape: (seq_len, features)
        # We can get features from feature_cols length
        n_features = len(feature_cols)
        input_shape = (seq_len, n_features)
        
        try:
            cnn_model = trainer.build_cnn(input_shape)
            cnn_callbacks = trainer.get_callbacks()
            
            cnn_model.fit(
                final_train_ds,
                validation_data=final_val_ds,
                epochs=cnn_epochs,
                callbacks=cnn_callbacks
            )
            trainer.save_cnn(cnn_model, "models/cnn_model.keras")
            print("CNN saved.")
        except Exception as e:
            print(f"CNN training failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping CNN training (No data or TF missing).")

if __name__ == "__main__":
    train_pipeline()
