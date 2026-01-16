# AI 外汇量化交易系统 (AI Forex Quantitative Trading System)

这是一个基于 Python 的模块化外汇量化交易系统，利用 LightGBM 和 1D-CNN 模型，结合 Yahoo Finance 历史数据和 TA-Lib 技术指标，进行市场趋势预测并提供交易信号。

## 🚀 功能特点

*   **数据获取**: 自动从 Yahoo Finance (`yfinance`) 获取外汇历史数据 (CSV格式)。
*   **特征工程**: 使用 TA-Lib 计算丰富技术指标 (MACD, RSI, BBANDS, ATR, OBV 等)。
*   **混合模型**: 结合传统机器学习 (LightGBM) 和深度学习 (1D-CNN) 进行趋势预测 (看涨/看跌/中性)。
*   **回测系统**: 向量化回测引擎，计算策略收益、夏普比率和最大回撤。
*   **实时 API**: 基于 FastAPI 的 REST 接口，提供实时交易信号。

---

## 🛠️ 环境依赖与安装

本项目在 Windows 环境下开发，推荐使用 Python 3.10+。

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd Gemini/2.0
```

### 2. 安装 TA-Lib (Windows 特别说明)
直接使用 pip 安装 TA-Lib 在 Windows 上通常会失败。请按照以下步骤操作：

1.  访问 [Christoph Gohlke 的 Python 扩展包下载页面](https://github.com/cgohlke/talib-build/releases) 或 [LFD UCI 镜像](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)。
2.  下载与您的 Python 版本和系统架构匹配的 `.whl` 文件。
    *   例如：Python 3.12, 64位系统 -> 下载 `TA_Lib‑0.4.28‑cp312‑cp312‑win_amd64.whl`
3.  本地安装 whl 文件：
    ```bash
    pip install "path/to/downloaded/TA_Lib‑0.4.xx‑cp3xx‑cp3xx‑win_amd64.whl"
    ```

### 3. 安装其他依赖
```bash
pip install -r requirements.txt
```

---

## ⚙️ 配置说明

### 1. 修改配置文件 (`config/config.yaml`)
您可以自定义交易对、时间周期和模型参数：
```yaml
# Data Configuration
# API_KEY not needed for yfinance

# 数据设置
# 数据设置
SYMBOLS:
  - "EURUSD"
  - "GBPUSD"
TIMEFRAME: "60min" # 时间周期: 15min, 30min, 60min, daily
DATA_SOURCE: "yfinance" # 选项: "yfinance" 或 "local_csv"
CSV_DATA_DIR: "data" # 本地 CSV 文件夹路径 (当 DATA_SOURCE 为 "local_csv" 时生效)
OUTPUT_DIR: "data"

# 模型参数
CNN_PARAMS:
  sequence_length: 60  # 回看窗口长度
  epochs: 50

### 2. 使用本地 CSV 数据
如果您想使用自己的数据，请将 `DATA_SOURCE` 设置为 `local_csv`，并将 CSV 文件放入 `CSV_DATA_DIR` 指定的目录。

**CSV 文件名**: `{SYMBOL}.csv` (例如 `EURUSD.csv`)

**CSV 格式示例** (必须包含 datetime 索引和 OHLC 列，不区分大小写):
```csv
Date,Open,High,Low,Close,Volume
2023-01-01 00:00:00,1.0700,1.0750,1.0680,1.0720,1000
2023-01-01 01:00:00,1.0720,1.0740,1.0710,1.0730,1200
```
*注意：如果没有 Volume 列，系统会自动填充为 0。*
```

---

## 🏃‍♂️ 快速开始 (使用教程)

### 步骤 1: 训练模型 (Train)
运行训练流水线，它将自动下载数据、预处理、训练 LightGBM 和 CNN 模型，并保存到 `models/` 目录。

```bash
python src/train_pipeline.py
# 注意：train_pipeline.py 实际文件名可能是 src/train.py，请查看目录下文件。
# 根据当前文件结构，运行：
python src/train.py
```
*输出：模型文件 (`lgbm_model.txt`, `cnn_model.keras`) 和 缩放器 (`scaler.pkl`) 将保存在 `models/` 文件夹中。*

### 步骤 2: 策略回测 (Backtest)
使用历史数据测试策略表现。

```bash
python src/backtest.py
```
*输出：将在 `data/` 目录下生成回测资金曲线图 (`EURUSD_backtest.png`)，并在控制台输出 ROI 和夏普比率。*

### 步骤 3: 启动 API 服务 (Inference)
启动本地预测服务，用于接入交易终端或前端展示。

```bash
uvicorn src.api:app --reload
```
*   服务启动后，访问文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   获取预测接口示例：
    ```
    GET http://127.0.0.1:8000/predict/EURUSD
    ```
    **响应示例**:
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

## 📂 项目结构

```
Gemini/2.0/
├── config/             # 配置文件
│   └── config.yaml
├── data/               # 存放下载的数据和回测结果图片
├── models/             # 存放训练好的模型 (LightGBM, CNN) 和 Scaler
├── notebooks/          # Jupyter Notebooks (用于 Colab 实验)
├── src/                # 源代码
│   ├── api.py          # FastAPI 接口服务
│   ├── backtest.py     # 回测引擎
│   ├── data_loader.py  # 数据加载与缓存
│   ├── models.py       # 模型定义 (Model Factory)
│   ├── preprocessing.py# 特征工程与数据预处理
│   ├── train.py        # 训练流水线入口
│   └── __init__.py
├── requirements.txt    # 项目依赖
├── README.md           # 项目文档
└── prompt.txt          # 原始需求描述
```
