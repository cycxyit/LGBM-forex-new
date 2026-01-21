import numpy as np
import lightgbm as lgb
from typing import Tuple

# Try importing tensorflow/keras, handle if not present (though required)
try:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, 
                                       BatchNormalization, Add, Activation, GlobalAveragePooling1D)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

    def _residual_block(self, x, filters, kernel_size=3, stride=1, l2_reg=0.0):
        shortcut = x
        
        # First Conv
        x = Conv1D(filters, kernel_size, strides=stride, padding='same', 
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # Second Conv
        x = Conv1D(filters, kernel_size, strides=1, padding='same', 
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization()(x)
        
        # Adjust shortcut if dimensions don't match
        if x.shape[-1] != shortcut.shape[-1]:
            shortcut = Conv1D(filters, 1, strides=stride, padding='same')(shortcut)
            
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def build_cnn(self, input_shape: Tuple[int, int], num_classes: int = 3):
        """
        Build ResNet-1D Model
        """
        if not HAS_TF:
            raise ImportError("TensorFlow not installed.")
            
        cnn_params = self.config.get("CNN_PARAMS", {})
        dropout_rate = cnn_params.get("dropout_rate", 0.3)
        l2_reg = cnn_params.get("l2_regularization", 0.001)
        
        inputs = Input(shape=input_shape)
        
        # Initial Conv
        x = Conv1D(64, 7, strides=2, padding='same', kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual Blocks
        x = self._residual_block(x, 64, l2_reg=l2_reg)
        x = self._residual_block(x, 64, l2_reg=l2_reg)
        
        x = self._residual_block(x, 128, stride=2, l2_reg=l2_reg)
        x = self._residual_block(x, 128, l2_reg=l2_reg)
        
        x = self._residual_block(x, 256, stride=2, l2_reg=l2_reg)
        x = self._residual_block(x, 256, l2_reg=l2_reg)
        
        # Global Pooling and Output
        x = GlobalAveragePooling1D()(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
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
                patience=10, # Increased patience for better convergence
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

    def save_lgbm(self, model: lgb.Booster, path: str):
        model.save_model(path)
        
    def save_cnn(self, model, path: str):
        model.save(path)
