import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
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
        
    def load_models(self):
        if os.path.exists("models/lgbm_model.txt"):
            self.lgbm_model = lgb.Booster(model_file="models/lgbm_model.txt")
            
        if os.path.exists("models/cnn_model.keras"):
             self.cnn_model = load_model("models/cnn_model.keras")

    def run_backtest(self, symbol: str, initial_balance: float = 10000.0):
        print(f"Backtesting {symbol}...")
        df = self.loader.fetch_data(symbol, interval=self.config.get("TIMEFRAME", "60min"))
        df = self.preprocessor.add_technical_indicators(df)
        df = self.preprocessor.create_labels(df)
        
        feature_cols = [c for c in df.columns if c not in ['target', 'date']]
        
        # Load scaler
        import pickle
        if not os.path.exists("models/scaler.pkl"):
            print("Scaler not found.")
            return
        with open("models/scaler.pkl", "rb") as f:
            self.preprocessor.scaler = pickle.load(f)

        # Prepare data
        X, y = self.preprocessor.prepare_for_training(df, feature_cols)
        
        # Predict (using LGBM for speed in example, can be ensemble)
        if self.lgbm_model:
            preds = self.lgbm_model.predict(X)
            signals = np.argmax(preds, axis=1) # 0: Bear, 1: Neu, 2: Bull
        else:
            print("Model not loaded.")
            return

        # Vectorised Backtest
        df['signal'] = signals
        
        # Calculate Strategy Returns
        # Shift signals by 1 because we trade on the 'next' candle based on 'current' signal
        df['position'] = 0
        df.loc[df['signal'] == 2, 'position'] = 1   # Long
        df.loc[df['signal'] == 0, 'position'] = -1  # Short
        
        # Returns
        df['market_returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['market_returns']
        
        # Cumulative Returns
        df['cumulative_market_returns'] = (1 + df['market_returns']).cumprod() * initial_balance
        df['cumulative_strategy_returns'] = (1 + df['strategy_returns']).cumprod() * initial_balance
        
        # Metrics
        total_return = df['cumulative_strategy_returns'].iloc[-1] - initial_balance
        roi = (total_return / initial_balance) * 100
        sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252 * 24) # approx hourly
        
        print(f"Initial Balance: {initial_balance}")
        print(f"Final Balance: {df['cumulative_strategy_returns'].iloc[-1]:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(df['cumulative_market_returns'], label='Buy & Hold')
        plt.plot(df['cumulative_strategy_returns'], label='AI Strategy')
        plt.title(f"Backtest Results for {symbol}")
        plt.legend()
        plt.grid()
        plt.savefig(f"data/{symbol}_backtest.png")
        print(f"Plot saved to data/{symbol}_backtest.png")
        
if __name__ == "__main__":
    backtester = Backtester()
    backtester.load_models()
    backtester.run_backtest("EURUSD")
