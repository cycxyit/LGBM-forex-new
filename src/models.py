import numpy as np
import lightgbm as lgb
from typing import Tuple

# Try importing tensorflow/keras, handle if not present (though required)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("Warning: TensorFlow/Keras not found. CNN model will fail.")

class ModelFactory:
    def __init__(self, config: dict):
        self.config = config
        
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                       X_val: np.ndarray, y_val: np.ndarray) -> lgb.Booster:
        """
        Train LightGBM model.
        """
        params = self.config.get("LIGHTGBM_PARAMS", {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbose": -1
        })
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            # num_boost_round=1000, 
            # early_stopping_rounds=50, # older lgb versions
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=10)
            ]
        )
        return model

    def build_cnn(self, input_shape: Tuple[int, int], num_classes: int = 3):
        """
        Build 1D-CNN Model
        """
        if not HAS_TF:
            raise ImportError("TensorFlow not installed.")
            
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def save_lgbm(self, model: lgb.Booster, path: str):
        model.save_model(path)
        
    def save_cnn(self, model, path: str):
        model.save(path)
