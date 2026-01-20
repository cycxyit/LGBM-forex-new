import pandas as pd
import numpy as np
try:
    import talib
except ImportError:
    talib = None

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        if talib is None:
            print("Warning: TA-Lib not found. Indicators will be limited or fail.")

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators using TA-Lib"""
        if df.empty:
            return df
        
        # Ensure data is sorted
        df = df.sort_index()

        # Prices
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        if talib:
            # Trend Indicators
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['SMA_50'] = talib.SMA(close, timeperiod=50)
            df['EMA_12'] = talib.EMA(close, timeperiod=12)
            df['EMA_26'] = talib.EMA(close, timeperiod=26)
            
            # Momentum
            df['RSI'] = talib.RSI(close, timeperiod=14)
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                close, fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # Volatility
            df['upper_band'], df['middle_band'], df['lower_band'] = talib.BBANDS(
                close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
            
            # Volume (if available, tick_volume for FX)
            # FX data usually has 'tick_volume' not 'volume'
            # Check column existence
            vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'
            if vol_col in df.columns:
                 df['OBV'] = talib.OBV(close, df[vol_col].astype(float).values)

        # Drop NaNs created by indicators (e.g., first 50 rows)
        df.dropna(inplace=True)
        return df

    def create_labels(self, df: pd.DataFrame, target_col: str = 'close', horizon: int = 1) -> pd.DataFrame:
        """
        Create target labels:
        0: Bearish (Price drops)
        1: Neutral (Price stays within small range - optional, using 2 classes for now to match user request of Up/Down prob)
        2: Bullish (Price rises)
        
        Wait, user asked for: Bullish Prob, Bearish Prob, Signal (Bull/Bear/Wait)
        So 3 classes is good.
        """
        # Simple Logic: 
        # Future Close > Current Close + threshold -> Bullish (2)
        # Future Close < Current Close - threshold -> Bearish (0)
        # Else -> Neutral (1)
        
        future_close = df[target_col].shift(-horizon)
        change = (future_close - df[target_col]) / df[target_col]
        
        # Threshold for neutrality (e.g., 0.05% change)
        threshold = 0.0005 
        
        conditions = [
            (change < -threshold),
            (abs(change) <= threshold),
            (change > threshold)
        ]
        choices = [0, 1, 2] # 0: Bear, 1: Neutral, 2: Bull
        
        df['target'] = np.select(conditions, choices, default=1)
        
        # Drop the last 'horizon' rows as they have no target
        df.dropna(subset=['target'], inplace=True)
        return df

    def prepare_for_training(self, df: pd.DataFrame, feature_cols: list, target_col: str = 'target', fit_scaler: bool = False):
        """
        Split into X and y.
        If fit_scaler is True, fit the scaler on X.
        Always transform X.
        """
        X = df[feature_cols].values
        y = df[target_col].values
        
        if fit_scaler:
            X = self.scaler.fit_transform(X)
        else:
            # Check if scaler is fitted
            try:
                X = self.scaler.transform(X)
            except Exception:
                # If not fitted, user might intend to fit, but let's warn or error
                # For this simple project, if not fitted and fit=False, maybe we should error.
                # But to be safe if user runs pipeline, we assume train handles it.
                if hasattr(self.scaler, 'mean_'): 
                     X = self.scaler.transform(X)
                else: 
                     # Fallback if uninitialized (should invoke with fit=True first)
                     raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
        
        return X, y

    def create_sequences(self, X, y, lookback: int = 60):
        """Create sequences for CNN/LSTM [samples, time_steps, features]"""
        Xs, ys = [], []
        for i in range(len(X) - lookback):
            Xs.append(X[i:(i + lookback)])
            ys.append(y[i + lookback])
        return np.array(Xs), np.array(ys)

    def create_tf_dataset(self, X, y, lookback: int = 60, batch_size: int = 32, shuffle: bool = False):
        """
        Create a tf.data.Dataset using timeseries_dataset_from_array.
        This generates sequences on-the-fly, saving huge amounts of RAM.
        """
        try:
            from tensorflow.keras.utils import timeseries_dataset_from_array
        except ImportError:
            raise ImportError("TensorFlow not installed.")
            
        # y needs to be aligned. 
        # timeseries_dataset_from_array produces batch of (X_seq, y_target)
        # It takes the target corresponding to the END of the sequence.
        # So we pass y starting from 'lookback' index?
        # No, the API default is: yields (data[i:i+seq_len], targets[i+seq_len]) if configured?
        # Actually API is: timeseries_dataset_from_array(data, targets, ...)
        # If we want target[i+lookback] associated with data[i:i+lookback],
        # we should treat 'targets' argument carefully.
        
        # Standard usage:
        # data: Array of shape (N, features)
        # targets: Array of shape (N,) or None. 
        # sequence_length: L
        # sequence_stride: 1
        # sampling_rate: 1
        # shuffle: bool
        
        # If targets is provided, the dataset yields (batch_data, batch_targets).
        # Where batch_data[j] = data[i:i+L]
        # And batch_targets[j] = targets[i] ?? No.
        # Documentation: "The targets corresponding to each sequence starting at index i... is targets[i]?"
        # Usually for "predict next step", we want target at time t+1 for window t-L..t
        
        # Let's align explicitly to be sure:
        # We want X[i : i+lookback] to predict y[i+lookback]
        # So we slice y to start at 'lookback'.
        # And we stop generating X when we run out of y.
        # But `timeseries_dataset_from_array` aligns by index.
        # If data[i] starts the sequence, targets[i] is the target? That's usually not what we want for "next step".
        # We want the target for the sequence ending at i+L.
        
        # Proper alignment:
        # Pass `targets` shifted by `lookback`?
        # Actually simpler: 
        # Just manually slice targets so they align with the START of the window?
        # If dataset yields Sequence[i] = data[i : i+L]
        # We want Target[i] to be y[i + L]
        # So we pass `targets = y[lookback:]`.
        # And we limit data to `data[:-lookback]`? 
        # Let's verify lengths.
        # len(y_sliced) = N - lookback.
        # We want N - lookback sequences.
        # Dataset will stop when it runs out of targets.
        
        # So:
        # data = X
        # targets = y[lookback:] (padded with something at start?)
        # Let's try:
        # targets = np.concatenate([np.zeros(lookback)*np.nan, y]) ? No that breaks types.
        
        # Correct approach with this API:
        # Use `end_index` parameter to limit where sequences start.
        # But alignment of target is key.
        # If we pass targets=y[lookback:], then index 0 of targets corresponds to index 0 of the dataset.
        # Index 0 of dataset is sequence starting at X[0].
        # Sequence X[0:L] -> predicts y[L].
        # Target[0] is y[lookback] -> y[L].
        # Match!
        
        dataset = timeseries_dataset_from_array(
            data=X,
            targets=y[lookback:], 
            sequence_length=lookback,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        
        # Explicitly set shapes to avoid "unknown shape" errors in Graph execution, 
        # especially for Flatten layers which need to know non-batch dims.
        # X shape: (None, lookback, n_features)
        # y shape: (None,)
        
        n_features = X.shape[1]
        
        def set_shapes(x, y):
            x.set_shape([None, lookback, n_features])
            y.set_shape([None])
            return x, y
            
        dataset = dataset.map(set_shapes)
        
        return dataset

if __name__ == "__main__":
    # Test stub
    pass
