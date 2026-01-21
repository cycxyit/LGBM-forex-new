import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import pickle
from datetime import datetime
try:
    from tensorflow.keras.models import load_model # types: ignore
except ImportError:
    pass

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.train import robust_load_config

class Backtester:
    def __init__(self, config_path="config/config.yaml"):
        self.config = robust_load_config(config_path)
        self.loader = DataLoader(config_path)
        self.preprocessor = DataPreprocessor()
        self.lgbm_model = None
        self.cnn_model = None
        
        # Load Backtest Config
        self.bt_config = self.config.get("BACKTEST", {})
        self.start_date = self.bt_config.get("START_DATE", "2023-01-01")
        self.end_date = self.bt_config.get("END_DATE", "2023-12-31")
        self.initial_balance = self.bt_config.get("INITIAL_BALANCE", 10000.0)
        self.transaction_cost = self.bt_config.get("TRANSACTION_COST", 0.00005)
        self.do_ensemble = self.bt_config.get("ENSEMBLE", True)
        
    def load_models(self):
        """Load trained models and scaler."""
        # Load Scaler (Strictly required)
        if not os.path.exists("models/scaler.pkl"):
            print("CRITICAL: Scaler not found at 'models/scaler.pkl'. Cannot proceed to prevent data leakage.")
            return False
            
        with open("models/scaler.pkl", "rb") as f:
            self.preprocessor.scaler = pickle.load(f)
        print("Scaler loaded successfully.")

        # Load LGBM
        if os.path.exists("models/lgbm_model.txt"):
            self.lgbm_model = lgb.Booster(model_file="models/lgbm_model.txt")
            print("LightGBM model loaded.")
        else:
            print("Warning: LightGBM model not found.")
            
        # Load CNN
        if os.path.exists("models/cnn_model.keras"):
            try:
                self.cnn_model = load_model("models/cnn_model.keras")
                print("CNN model loaded.")
            except Exception as e:
                print(f"Failed to load CNN model: {e}")
        else:
            print("Warning: CNN model not found.")
            
        return True

    def run_backtest(self, symbol: str):
        print(f"--- Starting Backtest for {symbol} ---")
        print(f"Period: {self.start_date} to {self.end_date}")
        
        # 1. Fetch Data
        df = self.loader.fetch_data(symbol, interval=self.config.get("TIMEFRAME", "60min"), period="2y")
        if df.empty:
            print("Error: No data fetched.")
            return

        # 2. Filter Data (Strict Time Separation)
        if df.index.tz is not None:
             df.index = df.index.tz_localize(None)
        
        mask = (df.index >= self.start_date) & (df.index <= self.end_date)
        df = df.loc[mask].copy()
        
        if df.empty:
            print(f"Error: No data found for specified date range {self.start_date} to {self.end_date}.")
            return
            
        print(f"Data loaded: {len(df)} candles.")

        # 3. Preprocessing
        # 3. Preprocessing
        # Lookback for indicators matching training
        df = self.preprocessor.add_technical_indicators(df)
        df = self.preprocessor.create_labels(df)
        
        # Drop NaN
        df.dropna(inplace=True)
        
        if df.empty:
            print("Error: DataFrame empty after preprocessing.")
            return

        # Prepare Features matching src/train.py logic
        # 1. LGBM Base Features (Tech Indicators)
        exclude_cols = ['target', 'date', 'open', 'high', 'low', 'close', 'volume', 'tick_volume', 'signal', 'position', 'strategy_returns', 'market_returns']
        lgbm_base_features = [c for c in df.columns if c not in exclude_cols]
        
        # Validate Scaler Features
        # Warning: If columns in backtest differ from training (e.g. order), this will fail or be wrong.
        # Ideally we save the feature name list.
        # For now, we assume same TA-Lib generation.
        
        X_lgbm_base = df[lgbm_base_features].values
        
        try:
            X_lgbm_base_scaled = self.preprocessor.scaler.transform(X_lgbm_base)
        except ValueError as e:
            print(f"Scaler Mismatch: {e}")
            print(f"Expected {self.preprocessor.scaler.n_features_in_} features, got {X_lgbm_base.shape[1]}")
            print(f"Columns: {lgbm_base_features}")
            return

        # 2. CNN Features (Raw OHLCV)
        cnn_features = ['open', 'high', 'low', 'close', 'volume']
        if 'tick_volume' in df.columns:
             cnn_features = ['open', 'high', 'low', 'close', 'tick_volume']
             
        seq_len = self.config.get("CNN_PARAMS", {}).get("sequence_length", 60)
        
        # We need alignment. 
        # CNN seq at index i uses data [i : i+seq_len].
        # LGBM needs features at index i+seq_len-1 (end of sequence).
        # We predict for target at i+seq_len.
        
        # Generate CNN sequences
        # Note: create_normalized_sequences returns y at i+seq_len
        # We don't need y for inference, but we need X.
        # We can use the same function and ignore y.
        
        X_cnn_seq, _ = self.preprocessor.create_normalized_sequences(
            df, 'target', lookback=seq_len, feature_cols=cnn_features
        )
        
        if len(X_cnn_seq) == 0:
            print("Not enough data for CNN sequences.")
            return
            
        # Align LGBM features
        # X_cnn_seq[0] corresponds to df index 0..seq_len-1.
        # Prediction is for next candle (feature vector at seq_len-1).
        # We need X_lgbm_base_scaled at indices correspoding to END of sequences.
        # If loop i goes 0..N, sequence is i..i+seq_len. End is i+seq_len-1.
        
        # Slicing: [seq_len-1 : ]
        # Length check:
        # X_cnn_seq length = len(df) - seq_len.
        # X_lgbm_base_scaled[seq_len-1 : -1] ?? 
        # In train.py we did [seq_len-1 : -1] because we had targets up to len-1.
        # Here we just want to predict for all available sequences.
        # Last sequence: i = len(df) - seq_len - 1. End = len(df) - 2.
        # Wait, create_normalized_sequences loop: range(len(X) - lookback).
        # Max i = len - lookback - 1.
        # Sequence: i .. i+lookback.
        # Xs[last] = data[len-lookback-1 : len-1].
        # Length of that is lookback.
        # End index in df is len-1.
        # So we need LGBM feature at index len-1.
        
        # Slice X_lgbm_base_scaled to match X_cnn_seq.
        # If X_cnn_seq[k] ends at index k + seq_len - 1.
        # We want X_lgbm_base_scaled[k + seq_len - 1].
        # So slice X_lgbm_base_scaled from seq_len-1 to end.
        
        start_idx = seq_len - 1
        X_lgbm_aligned = X_lgbm_base_scaled[start_idx : start_idx + len(X_cnn_seq)]
        
        # Verify
        if len(X_lgbm_aligned) != len(X_cnn_seq):
             # This happens if integer math is slightly off or specific loop range
             # create_normalized_sequences len = N - seq_len.
             # start_idx = seq_len - 1.
             # Remaining = N - (seq_len - 1) = N - seq_len + 1.
             # One extra? 
             # Let's truncate to min length.
             min_len = min(len(X_lgbm_aligned), len(X_cnn_seq))
             X_lgbm_aligned = X_lgbm_aligned[:min_len]
             X_cnn_seq = X_cnn_seq[:min_len]

        # 4. Hybrid Prediction
        final_probs = None
        
        if self.cnn_model and self.lgbm_model:
            # a) Extract CNN Embeddings
            # We need the feature extractor part of the model
            # Re-build or get layer? 
            # In backtest we loaded the full model.
            try:
                from tensorflow.keras.models import Model
                feature_layer = self.cnn_model.get_layer('feature_dense')
                feature_extractor = Model(inputs=self.cnn_model.input, outputs=feature_layer.output)
                
                cnn_embeddings = feature_extractor.predict(X_cnn_seq, verbose=0)
                
                # b) Fuse
                X_final = np.hstack([X_lgbm_aligned, cnn_embeddings])
                
                # c) Predict LGBM
                final_probs = self.lgbm_model.predict(X_final)
                
            except Exception as e:
                print(f"Hybrid prediction failed: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            print("Models not loaded correctly for Hybrid mode.")
            return

        # Align signals back to DataFrame
        # We lost `start_idx` rows at the start.
        # Fill them with 0 or NaNs.
        
        full_probs = np.zeros((len(df), 3))
        # Fill the tail
        # slice: [start_idx : start_idx + len]
        full_probs[start_idx : start_idx + len(final_probs)] = final_probs
        
        signals = np.argmax(full_probs, axis=1)
        # Force 0/1 signal to 0 for the warm-up period
        signals[:start_idx] = 1 # Neutral/Hold

        
        # 5. Calculate Returns
        df['signal'] = signals
        
        # Shift signal by 1: We trade at Open of Next candle based on Signal from Close of Current
        df['position'] = 0
        df.loc[df['signal'] == 2, 'position'] = 1   # Long
        df.loc[df['signal'] == 0, 'position'] = -1  # Short
        
        # Calculate Cost
        # Position change: abs(curr - prev). 0->1 = 1, 1->-1 = 2, etc.
        df['pos_change'] = df['position'].diff().abs().fillna(0)
        df['cost'] = df['pos_change'] * self.transaction_cost
        
        # Returns
        df['market_returns'] = df['close'].pct_change()
        
        # Strategy Return = Position(t-1) * MarketReturn(t) - Cost(t)
        # We pay cost when we ENTER/EXIT. 
        # Approx: Deduct cost from return.
        df['strategy_returns'] = (df['position'].shift(1) * df['market_returns']) - df['cost']
        
        # Cumulative
        df['cumulative_market_returns'] = (1 + df['market_returns']).cumprod() * self.initial_balance
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() * self.initial_balance
        
        # 6. Metrics
        total_return = df['cumulative_strategy_returns'].iloc[-1] - self.initial_balance
        roi = (total_return / self.initial_balance) * 100
        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 24) if df['strategy_returns'].std() > 0 else 0
        max_drawdown = (df['cumulative_strategy_returns'] / df['cumulative_strategy_returns'].cummax() - 1).min() * 100
        
        print(f"\n--- Results for {symbol} ---")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance:   ${df['cumulative_strategy_returns'].iloc[-1]:,.2f}")
        print(f"Net Profit:      ${total_return:,.2f}")
        print(f"ROI:             {roi:.2f}%")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Max Drawdown:    {max_drawdown:.2f}%")
        print(f"Trades Executed: {int(df['pos_change'].sum())}")
        
        # 7. Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['cumulative_market_returns'], label='Buy & Hold', alpha=0.6)
        plt.plot(df.index, df['cumulative_strategy_returns'], label='AI Strategy (Net)', linewidth=2)
        plt.title(f"Backtest: {symbol} | ROI: {roi:.2f}% | Costs: Included")
        plt.xlabel("Date")
        plt.ylabel("Balance ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = f"data/{symbol}_backtest.png"
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    backtester = Backtester()
    if backtester.load_models():
        backtester.run_backtest("EURUSD")
