import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Dict
import yfinance as yf
import time

class DataLoader:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        # self.api_key = self._get_api_key() # Not needed for yfinance
        self.output_dir = Path(self.config.get("OUTPUT_DIR", "data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # _get_api_key removed as it is not needed

    def fetch_data(self, symbol: str, interval: str = "60min", period: str = "2y") -> pd.DataFrame:
        """
        Fetch forex data from Yahoo Finance.
        
        Args:
            symbol: Currency pair (e.g., EURUSD=X) - yfinance usually needs =X for forex
            interval: Time interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
                      Note: yfinance '60min' is usually '1h'
            period: Data period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            pd.DataFrame with columns [open, high, low, close, volume] indexed by timestamp
            Columns are lowercased to match previous behavior.
        """
        # Map config interval to yfinance interval if necessary
        interval_map = {
            "60min": "1h",
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "daily": "1d"
        }
        yf_interval = interval_map.get(interval, interval)
        
        # Ensure symbol format for Forex in yfinance (usually ends with =X)
        if len(symbol) == 6 and not symbol.endswith("=X"):
             # Assuming standard forex 6 char symbol like EURUSD
             ticker_symbol = f"{symbol}=X"
        else:
             ticker_symbol = symbol

        print(f"Fetching data for {ticker_symbol} ({interval})...")
        
        # Check cache first
        cache_file = self.output_dir / f"{symbol}_{interval}.csv"
        # Optional: Check if cache is fresh enough? For now, we rely on manual deletion or overwrite if forced.
        # But to be consistent with previous logic, if cache exists, we might want to load it
        # However, usually we want fresh data. Let's stick to previous behavior: if cache exists, load it.
        # But wait, previous behavior checked cache first.
        if cache_file.exists():
             print(f"Loading from cache: {cache_file}")
             df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
             return df

        try:
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period=period, interval=yf_interval)
            
            if df.empty:
                print(f"Warning: No data found for {ticker_symbol}")
                return pd.DataFrame()

            # Clean up DataFrame
            # yfinance returns: Open, High, Low, Close, Volume, Dividends, Stock Splits
            # We only need OHLCV and we want lowercase columns
            df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }, inplace=True)
            
            # Keep only necessary columns
            cols_to_keep = ["open", "high", "low", "close", "volume"]
            # Filter columns that actually exist (Volume might be missing for some forex)
            cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[cols_to_keep]
            
            # Ensure index is timezone naive or consistent if needed? 
            # yfinance returns timezone-aware index.
            # Let's standardise to tz-naive UTC for safety if needed, but often leaving it is fine until it hits a database.
            # safe option: convert to UTC and remove tz info
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Save to cache
            df.to_csv(cache_file)
            print(f"Saved to {cache_file}")
            
            return df

        except Exception as e:
            print(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()

    def get_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols in config"""
        symbols = self.config.get("SYMBOLS", [])
        interval = self.config.get("TIMEFRAME", "60min")
        
        data_map = {}
        for sym in symbols:
            df = self.fetch_data(sym, interval)
            if not df.empty:
                data_map[sym] = df
            
            # yfinance is lenient, but let's be polite
            time.sleep(1)
            
        return data_map

if __name__ == "__main__":
    # Test run
    loader = DataLoader()
    try:
        data = loader.get_all_symbols()
        for sym, df in data.items():
            print(f"{sym}: {df.shape}")
            print(df.head())
    except Exception as e:
        print(f"Error: {e}")

