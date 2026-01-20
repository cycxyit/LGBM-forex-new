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
        # Lookback for indicators matching training
        # Note: Indicators might need previous data. Ideally we load more data and then cut, 
        # but for simplicity we assume start_date has enough buffer or we accept small loss at start.
        df = self.preprocessor.add_technical_indicators(df)
        df = self.preprocessor.create_labels(df)
        
        # Drop NaN
        df.dropna(inplace=True)
        
        if df.empty:
            print("Error: DataFrame empty after preprocessing.")
            return

        feature_cols = [c for c in df.columns if c not in ['target', 'date', 'signal', 'position', 'strategy_returns', 'market_returns']]
        X, y = self.preprocessor.prepare_for_training(df, feature_cols, fit_scaler=False)
        
        # 4. Generate Predictions
        lgbm_probs = None
        cnn_probs = None
        
        if self.lgbm_model:
            lgbm_probs = self.lgbm_model.predict(X) 
            
        if self.cnn_model:
            seq_len = self.config.get("CNN_PARAMS", {}).get("sequence_length", 60)
            if len(X) > seq_len:
                Xs, _ = self.preprocessor.create_sequences(X, y, lookback=seq_len)
                # Padding for the first `seq_len` rows where CNN can't predict
                # We align predictions to the END of the sequence
                cnn_preds = self.cnn_model.predict(Xs, verbose=0)
                
                # Create full buffer for alignment
                cnn_probs = np.zeros((len(X), 3))
                cnn_probs[seq_len:] = cnn_preds
            else:
                 print("Warning: Not enough data for CNN sequence length.")

        # Ensemble Logic
        final_probs = None
        
        if self.do_ensemble and lgbm_probs is not None and cnn_probs is not None:
            # Simple Average
            final_probs = (lgbm_probs + cnn_probs) / 2
            # Zero out the first `seq_len` of ensemble since CNN was 0
            # Alternatively, fallback to LGBM for first rows
            final_probs[:60] = lgbm_probs[:60] 
        elif lgbm_probs is not None:
            final_probs = lgbm_probs
        elif cnn_probs is not None:
            final_probs = cnn_probs
        else:
            print("Critical: No models available for prediction.")
            return

        signals = np.argmax(final_probs, axis=1)
        
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
