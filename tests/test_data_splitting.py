import sys
import os
import numpy as np
import pandas as pd
import unittest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import DataPreprocessor

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        # Create dummy OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        self.df = pd.DataFrame({
            'open': np.linspace(100, 200, 100),
            'high': np.linspace(101, 201, 100),
            'low': np.linspace(99, 199, 100),
            'close': np.linspace(100, 200, 100), # Linear trend
            'volume': np.full(100, 1000),
            'target': np.zeros(100) # Dummy target
        }, index=dates)

    def test_local_normalization(self):
        """Test if sequences are normalized by dividing by the first close price."""
        lookback = 10
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        
        X, y = self.preprocessor.create_normalized_sequences(
            self.df, 'target', lookback=lookback, feature_cols=feature_cols
        )
        
        # Check shape
        # N = 100 - 10 = 90 sequences
        self.assertEqual(len(X), 90)
        self.assertEqual(X.shape, (90, 10, 5))
        
        # Check content of first sequence
        # Raw first seq: close from 100 to 110 (approx)
        # Base price = 100
        # Normalized first element should be close[0]/100 = 1.0 (if close is 4th col)
        
        # 'open' is 0, 'close' is 3
        # First row of first sequence: 
        # Open=100, High=101, Low=99, Close=100. Base=100.
        # Norm: 1.0, 1.01, 0.99, 1.0
        
        seq0 = X[0]
        self.assertAlmostEqual(seq0[0, 3], 1.0, places=5) # Close normalized to 1.0
        self.assertAlmostEqual(seq0[0, 0], 1.0, places=5) # Open normalized to 1.0
        
        # Volume (index 4)
        # Volume is constant 1000. 
        # Base vol = 1000. Norm = 1000/1000 = 1.0
        self.assertAlmostEqual(seq0[0, 4], 1.0, places=5)

    def test_strict_time_splitting(self):
        """Test manual splitting logic to ensure no overlap."""
        # Split index at 80
        split_idx = 80
        lookback = 10
        
        train_df = self.df.iloc[:split_idx]
        val_df = self.df.iloc[split_idx:]
        
        # Create sequences
        X_train, y_train = self.preprocessor.create_normalized_sequences(
            train_df, 'target', lookback=lookback, feature_cols=['close']
        )
        X_val, y_val = self.preprocessor.create_normalized_sequences(
            val_df, 'target', lookback=lookback, feature_cols=['close']
        )
        
        # Train should have 80 - 10 = 70 sequences
        self.assertEqual(len(X_train), 70)
        
        # Val should have 20 - 10 = 10 sequences
        # Note: In strict splitting, you lose the first 'lookback' samples of validation 
        # if you don't allow overlap. But 'suggest.md' says "val sequence strictly after train".
        # So val_df starting at 80 means first val seq uses 80..89 to predict 90.
        # Last train seq uses 69..79 to predict 80.
        # No overlap in *prediction target*? Or no overlap in *input data*?
        
        # Usually:
        # Time: 0.............80.............100
        # Train Input: [0..79]
        # Val Input: [80..99]
        # Overlap check:
        # Train Last Seq: Input 70..79 -> Target 80 (Wait, target[i+lookback])
        # If train_df is 0..80 (exclusive? iloc[:80] is 0..79)
        # create_sequences loop: range(len=80 - 10 = 70). i=0..69.
        # Last i=69. Seq: 69..79. Target: 79+10? No target is y[i+lookback] inside function?
        # Function: ys.append(y[i+lookback])
        # So target comes from within the df.
        # If train_df excludes index 80, then max index is 79.
        # create_seqs(train_df): Max target index is 79.
        
        # Val_df starts at 80.
        # create_seqs(val_df): First input 80..89. Target 90.
        
        # Gap?
        # Train predicts up to 79.
        # Val predicts starting at 90.
        # We miss predictions for 80..89 because Val input needs 10 steps of history.
        # If we want to predict 80, we need input 70..79.
        # But 70..79 is in Train DF.
        
        # Ideally: Val set should include lookback period from Train set if we want continuous predictions?
        # But for *training*, strict separation is safer. Gaps are acceptable in training.
        # "suggest.md": "确保 val 序列完全在 split_idx 之后" implies X_val depends only on data >= split_idx.
        
        self.assertEqual(len(X_val), 10)
        
        # Verify indices conceptually
        # Train data max index used: 79
        # Val data min index used: 80
        pass

if __name__ == '__main__':
    unittest.main()
