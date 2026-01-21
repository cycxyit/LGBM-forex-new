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
        Build Simplified CNN for Feature Extraction (Functional API)
        """
        if not HAS_TF:
            raise ImportError("TensorFlow not installed.")
            
        cnn_params = self.config.get("CNN_PARAMS", {})
        dropout_rate = cnn_params.get("dropout_rate", 0.3)
        l2_reg = cnn_params.get("l2_regularization", 0.001)
        
        # Functional API for easy layer access
        inputs = Input(shape=input_shape)
        
        # 1. Feature Extraction Blocks
        x = Conv1D(32, kernel_size=5, activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_reg))(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # 2. Global Aggregation
        x = GlobalAveragePooling1D()(x)
        
        # 3. Dense Feature Layer (The Embedding)
        # Naming this layer makes it easier to extract later if needed by name, 
        # though we can also just cut the model.
        features = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg), name='feature_dense')(x)
        
        # 4. Classification Head
        x = Dropout(dropout_rate)(features)
        outputs = Dense(num_classes, activation='softmax', name='prediction')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def get_feature_extractor(self, model):
        """
        Returns a new model that outputs the embeddings from the 'feature_dense' layer.
        """
        if not HAS_TF:
            return None
        
        try:
            # Try getting by name
            feature_layer = model.get_layer('feature_dense')
            return Model(inputs=model.input, outputs=feature_layer.output)
        except ValueError:
            # Fallback: assume it's the 3rd to last layer (before Dropout and Softmax) or similar
            # But relying on structure is brittle. Since we built it, we know the name.
            print("Could not find 'feature_dense' layer. Returning model as is (verify architecture).")
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
