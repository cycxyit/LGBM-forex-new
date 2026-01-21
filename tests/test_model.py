import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import ModelFactory

def test_cnn_build():
    print("Testing CNN Model Build...")
    
    config = {
        "CNN_PARAMS": {
            "dropout_rate": 0.3,
            "l2_regularization": 0.001
        }
    }
    
    factory = ModelFactory(config)
    
    # Simulate input shape (seq_len=60, features=10)
    input_shape = (60, 10)
    num_classes = 3
    
    try:
        model = factory.build_cnn(input_shape, num_classes)
        model.summary()
        print("\nModel built successfully!")
        
        # Test a dummy forward pass
        dummy_input = np.random.random((1, 60, 10))
        output = model.predict(dummy_input)
        print(f"\nDummy prediction shape: {output.shape}")
        
    except Exception as e:
        print(f"FAILED to build model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cnn_build()
