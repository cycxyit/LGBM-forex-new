import numpy as np
import lightgbm as lgb
from typing import Tuple

# Try importing tensorflow/keras, handle if not present (though required)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping
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
            
        cnn_params = self.config.get("CNN_PARAMS", {})
        dropout_rate = cnn_params.get("dropout_rate", 0.2)
        l2_reg = cnn_params.get("l2_regularization", 0.0)
        
        model = Sequential([
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    
    def get_callbacks(self):
        """Get training callbacks including EarlyStopping"""
        if not HAS_TF:
            return []
            
        return [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]

    def save_lgbm(self, model: lgb.Booster, path: str):
        model.save_model(path)
        
    def save_cnn(self, model, path: str):
        model.save(path)
