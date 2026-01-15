# AI Forex Quantitative Trading System

This is a modular AI-driven quantitative trading system for Forex markets based on Python. It utilizes LightGBM and 1D-CNN models, combined with Yahoo Finance historical data and TA-Lib technical indicators, to predict market trends and provide trading signals.

## 🚀 Features

*   **Data Acquisition**: Automatically fetch historical Forex data (CSV format) from Yahoo Finance (`yfinance`).
*   **Feature Engineering**: Calculate rich technical indicators (MACD, RSI, BBANDS, ATR, OBV, etc.) using TA-Lib.
*   **Hybrid Modeling**: Combine traditional machine learning (LightGBM) and deep learning (1D-CNN) for trend prediction (Bullish/Bearish/Neutral).
*   **Backtesting System**: Vectorized backtesting engine to calculate strategy returns, Sharpe ratio, and maximum drawdown.
*   **Real-time API**: REST interface based on FastAPI to provide real-time trading signals.

---

## 🛠️ Requirements & Installation

This project is developed in a Windows environment. Python 3.10+ is recommended.

### 1. Clone the Project
```bash
git clone <your-repo-url>
cd Gemini/2.0
```

### 2. Install TA-Lib (Windows Specific)
Directly installing TA-Lib via pip on Windows often fails. Please follow these steps:

1.  Visit [Christoph Gohlke's Python Extension Packages](https://github.com/cgohlke/talib-build/releases) or the [LFD UCI Mirror](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib).
2.  Download the `.whl` file that matches your Python version and system architecture.
    *   Example: For Python 3.12, 64-bit system -> Download `TA_Lib‑0.4.28‑cp312‑cp312‑win_amd64.whl`
3.  Install the whl file locally:
    ```bash
    pip install "path/to/downloaded/TA_Lib‑0.4.xx‑cp3xx‑cp3xx‑win_amd64.whl"
    ```

### 3. Install Other Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### 1. Modify Config (`config/config.yaml`)
You can customize trading pairs, timeframes, and model parameters:
```yaml
# Data Configuration
# API_KEY not needed for yfinance

# Data Settings
SYMBOLS:
  - "EURUSD"
  - "GBPUSD"
TIMEFRAME: "60min" # Timeframe: 15min, 30min, 60min, daily
OUTPUT_DIR: "data"

# Model Parameters
CNN_PARAMS:
  sequence_length: 60  # Lookback window length
  epochs: 50
```

---

## 🏃‍♂️ Quick Start (Tutorial)

### Step 1: Train Models
Run the training pipeline. It will automatically download data, preprocess it, train LightGBM and CNN models, and save them to the `models/` directory.

```bash
python src/train_pipeline.py
# Note: The actual filename might be src/train.py. Check the directory.
# Based on current structure, run:
python src/train.py
```
*Output: Model files (`lgbm_model.txt`, `cnn_model.keras`) and scaler (`scaler.pkl`) will be saved in the `models/` folder.*

### Step 2: Backtest Strategy
Test strategy performance using historical data.

```bash
python src/backtest.py
```
*Output: A backtest equity curve plot (`EURUSD_backtest.png`) will be generated in `data/`, and ROI/Sharpe Ratio will be printed to the console.*

### Step 3: Start API Service (Inference)
Start the local prediction service for integration with trading terminals or frontends.

```bash
uvicorn src.api:app --reload
```
*   Once started, access the docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   Prediction endpoint example:
    ```
    GET http://127.0.0.1:8000/predict/EURUSD
    ```
    **Response Example**:
    ```json
    {
      "symbol": "EURUSD",
      "predictions": {
        "lightgbm": {
          "bearish_prob": 0.1,
          "neutral_prob": 0.2,
          "bullish_prob": 0.7,
          "signal": "Bullish"
        },
        "cnn": { ... },
        "ensemble": {
          "signal": "Bullish"
        }
      }
    }
    ```

---

## 📂 Project Structure

```
Gemini/2.0/
├── config/             # Configuration files
│   └── config.yaml
├── data/               # Downloaded data and backtest plots
├── models/             # Trained models (LightGBM, CNN) and Scaler
├── notebooks/          # Jupyter Notebooks (for Colab experiments)
├── src/                # Source code
│   ├── api.py          # FastAPI service
│   ├── backtest.py     # Backtest engine
│   ├── data_loader.py  # Data loading and caching
│   ├── models.py       # Model definitions (Model Factory)
│   ├── preprocessing.py# Feature engineering and preprocessing
│   ├── train.py        # Training pipeline entry point
│   └── __init__.py
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation (Chinese)
├── README_en.md        # Project documentation (English)
└── prompt.txt          # Original requirement description
```
