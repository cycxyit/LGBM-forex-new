import yaml
import numpy as np
import pandas as pd
import pickle
import gc
from pathlib import Path
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import ModelFactory, HAS_TF

def robust_load_config(path: str = "config/config.yaml") -> dict:
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            "SYMBOLS": ["EURUSD"],
            "TIMEFRAME": "60min",
            "LIGHTGBM_PARAMS": {"objective": "multiclass", "num_class": 3, "verbose": -1},
            "CNN_PARAMS": {"sequence_length": 60, "epochs": 20, "batch_size": 32}
        }

def train_pipeline():
    print("Starting Hybrid CNN-LGBM Training Pipeline...")
    config = robust_load_config()
    
    # Initialize
    loader = DataLoader()
    preprocessor = DataPreprocessor()
    trainer = ModelFactory(config)
    
    symbols = config.get("SYMBOLS", ["EURUSD"])
    interval = config.get("TIMEFRAME", "60min")
    cnn_params = config.get("CNN_PARAMS", {})
    seq_len = cnn_params.get("sequence_length", 60)
    batch_size = cnn_params.get("batch_size", 32)
    cnn_epochs = cnn_params.get("epochs", 20)
    
    # 1. Load and Preprocess Data per Symbol
    # We will combine all symbols for training
    full_data = [] # List of (df, symbol)
    
    print("Step 1: Fetching Data...")
    for symbol in symbols:
        df = loader.fetch_data(symbol, interval=interval)
        if df.empty:
            continue
        
        # Add Technical Indicators (For LGBM)
        df = preprocessor.add_technical_indicators(df)
        
        # Create Labels (For both)
        df = preprocessor.create_labels(df, horizon=1)
        
        if len(df) > seq_len + 100: # Minimum size check
            full_data.append(df)
        else:
            print(f"Skipping {symbol}: Not enough data.")

    if not full_data:
        print("No valid data loaded.")
        return

    # 2. Strict Time-Based Splitting & Scaler Fitting
    # Strategy: 
    # - Split each symbol's timeline into Train (80%) and Val (20%).
    # - Fit Scaler only on Train parts.
    
    print("Step 2: Splitting and Scaling...")
    train_dfs = []
    val_dfs = []
    
    # Identify Feature Columns
    # CNN: Raw OHLCV
    cnn_features = ['open', 'high', 'low', 'close', 'volume']
    if 'tick_volume' in full_data[0].columns:
         cnn_features = ['open', 'high', 'low', 'close', 'tick_volume']
         
    # LGBM: Tech Indicators + Time features?
    # Exclude targets and date
    exclude_cols = ['target', 'date', 'open', 'high', 'low', 'close', 'volume', 'tick_volume']
    lgbm_base_features = [c for c in full_data[0].columns if c not in exclude_cols]
    
    # Fit Scaler on Concatenated Train Data
    # We need to collect all train data first
    all_train_features = []
    
    for df in full_data:
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        train_dfs.append(train_df)
        val_dfs.append(val_df)
        
        all_train_features.append(train_df[lgbm_base_features].values)
        
    # Fit Scaler (Global for LGBM features)
    preprocessor.scaler.fit(np.concatenate(all_train_features))
    del all_train_features
    
    # Save Scaler
    Path("models").mkdir(exist_ok=True)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(preprocessor.scaler, f)
        
    print(f"Scaler fitted. LGBM Base Features: {len(lgbm_base_features)}")
    
    # 3. Prepare CNN Datasets & Arrays
    print("Step 3: Preparing CNN Data (Local Normalization)...")
    
    X_cnn_train = []
    y_cnn_train = []
    X_cnn_val = []
    y_cnn_val = []
    
    # Also keep track of indices to align LGBM
    # We will generate arrays for LGBM in Step 5
    
    for df in train_dfs:
        X_seq, y_seq = preprocessor.create_normalized_sequences(
            df, 'target', lookback=seq_len, feature_cols=cnn_features
        )
        if len(X_seq) > 0:
            X_cnn_train.append(X_seq)
            y_cnn_train.append(y_seq)
            
    for df in val_dfs:
        X_seq, y_seq = preprocessor.create_normalized_sequences(
            df, 'target', lookback=seq_len, feature_cols=cnn_features
        )
        if len(X_seq) > 0:
            X_cnn_val.append(X_seq)
            y_cnn_val.append(y_seq)
            
    if not X_cnn_train:
        print("Error: No training data generated.")
        return
        
    X_cnn_train = np.concatenate(X_cnn_train)
    y_cnn_train = np.concatenate(y_cnn_train)
    X_cnn_val = np.concatenate(X_cnn_val)
    y_cnn_val = np.concatenate(y_cnn_val)
    
    print(f"CNN Train Shape: {X_cnn_train.shape}, Val Shape: {X_cnn_val.shape}")
    
    # 4. Train CNN (Feature Extractor)
    if HAS_TF:
        print("Step 4: Training CNN Feature Extractor...")
        input_shape = (seq_len, X_cnn_train.shape[2])
        cnn_model = trainer.build_cnn(input_shape, num_classes=3)
        
        # Use simple FIT for now (converting to Dataset internally or using array fit)
        # For large data, we used Generators, but robust fit is okay for moderate size
        callbacks = trainer.get_callbacks()
        
        cnn_model.fit(
            X_cnn_train, y_cnn_train,
            validation_data=(X_cnn_val, y_cnn_val),
            epochs=cnn_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        trainer.save_cnn(cnn_model, "models/cnn_model.keras")
        
        # 4b. Extract Features
        print("Extracting CNN Features...")
        feature_extractor = trainer.get_feature_extractor(cnn_model)
        
        # Get embeddings
        # We need embeddings for BOTH Train and Val to train LGBM
        # Note: We must ensure alignment matches X_cnn_train/val
        
        cnn_feat_train = feature_extractor.predict(X_cnn_train, batch_size=batch_size)
        cnn_feat_val = feature_extractor.predict(X_cnn_val, batch_size=batch_size)
        
        print(f"CNN Embeddings shape: {cnn_feat_train.shape}")
        
    else:
        print("TensorFlow missing. Skipping CNN Phase.")
        return

    # 5. Prepare LGBM Data (Align with CNN)
    print("Step 5: preparing LGBM Data & Fusion...")
    
    def get_aligned_lgbm_features(dfs_list):
        features_list = []
        targets_list = []
        
        for df in dfs_list:
            # We need to take rows corresponding to the END of sequences
            # sequence[i] uses indices i..i+seq_len-1
            # We want LGBM features at i+seq_len-1
            # And target at i+seq_len
            
            # create_normalized_sequences returns y at i+seq_len.
            # So targets are aligned.
            
            # We need features from df.iloc[seq_len-1 : -1] ??
            # Loops i=0 to len-seq_len
            # Last seq index = (len-seq_len-1) + seq_len-1 = len-2
            # So features range: index (seq_len-1) to (len-2)
            
            # Verify lengths: N = len(df) - seq_len
            # feature slice length: (len-2) - (seq_len-1) + 1 = len - seq_len. Matches N.
            
            # Transform Features first
            X_base = df[lgbm_base_features].values
            X_base_scaled = preprocessor.scaler.transform(X_base)
            
            # Slice
            # Indices: seq_len-1, seq_len, ..., len-2
            # Slice: [seq_len-1 : -1]
            # (Note: -1 index is the LAST element, usually we want up to len-1 exclusive?
            # No, we want UP TO len-2 (inclusive).
            # Python slice [start : end_exclusive].
            # Target indices were i+seq_len. Max i = len-seq_len-1. Max Target idx = len-1.
            # Corresponding feature idx = len-2. (One step before target).
            # Slice [seq_len-1 : -1] excludes the last element (len-1). So it ends at len-2. Correct.
            
            X_aligned = X_base_scaled[seq_len-1 : -1]
            features_list.append(X_aligned)
            
        return np.concatenate(features_list)

    X_lgbm_base_train = get_aligned_lgbm_features(train_dfs)
    X_lgbm_base_val = get_aligned_lgbm_features(val_dfs)
    
    # Verify Shapes
    assert len(X_lgbm_base_train) == len(cnn_feat_train), f"Shape mismatch: LGBM {len(X_lgbm_base_train)} vs CNN {len(cnn_feat_train)}"
    
    # FUSION
    X_final_train = np.hstack([X_lgbm_base_train, cnn_feat_train])
    X_final_val = np.hstack([X_lgbm_base_val, cnn_feat_val])
    
    # Targets are already aggregated in step 3
    y_final_train = y_cnn_train
    y_final_val = y_cnn_val
    
    print(f"Final Training Data Shape: {X_final_train.shape}")
    
    # 6. Train LGBM
    print("Step 6: Training LightGBM...")
    
    lgbm_model = trainer.train_lightgbm(
        X_final_train, y_final_train,
        X_final_val, y_final_val
    )
    
    trainer.save_lgbm(lgbm_model, "models/lgbm_model.txt")
    print("Hybrid Training Complete!")

if __name__ == "__main__":
    try:
        train_pipeline()
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
