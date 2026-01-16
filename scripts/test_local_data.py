import pandas as pd
import yaml
from src.data_loader import DataLoader
import os

# Create a temporary config for testing
test_config = {
    "DATA_SOURCE": "local_csv",
    "CSV_DATA_DIR": "data",
    "OUTPUT_DIR": "data",
    "SYMBOLS": ["TEST_SYM"],
    "TIMEFRAME": "60min"
}

with open("config/test_config.yaml", "w") as f:
    yaml.dump(test_config, f)

print("Testing Local CSV Loader...")
loader = DataLoader(config_path="config/test_config.yaml")
df = loader.fetch_data("TEST_SYM")

if not df.empty:
    print(f"Success! Loaded {len(df)} rows.")
    print(df.head())
    if "open" in df.columns and "volume" in df.columns:
         print("Columns validated.")
    else:
         print("Column validation failed.")
else:
    print("Failed to load data.")

# Clean up
os.remove("config/test_config.yaml")
