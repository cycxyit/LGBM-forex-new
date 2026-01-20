# AI 外汇量化交易系统 (AI Forex Quantitative Trading System)

这是一个基于 Python 的模块化外汇量化交易系统，利用 LightGBM 和 1D-CNN 模型，结合 Yahoo Finance 历史数据和 TA-Lib 技术指标，进行市场趋势预测并提供交易信号。

## 🚀 功能特点

*   **数据获取**: 自动从 Yahoo Finance (`yfinance`) 获取外汇历史数据 (CSV格式)。
*   **特征工程**: 使用 TA-Lib 计算丰富技术指标 (MACD, RSI, BBANDS, ATR, OBV 等)。
*   **混合模型**: 结合传统机器学习 (LightGBM) 和深度学习 (1D-CNN) 进行趋势预测 (看涨/看跌/中性)。
*   **防数据泄露回测**: 
    - **严格时间分割**: 强制指定回测起止日期，确保不使用训练数据的未来信息。
    - **Scaler 隔离**: 强制使用训练阶段保存的标准化器，禁止测试集重新拟合。
*   **真实成本模拟**: 支持设置点差/手续费（默认 0.5 pips），包含成本的净值曲线更贴近实盘。
*   **模型集成**: 支持同时使用 LightGBM 和 CNN 预测结果取平均，提高稳健性。
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
您可以在此文件中全量配置交易系统：

```yaml
# Data Configuration
SYMBOLS:
  - "EURUSD"
  - "GBPUSD"
TIMEFRAME: "60min" # 时间周期: 1min, 5min, 15min, 30min, 60min, daily
DATA_SOURCE: "yfinance" # 选项: "yfinance" 或 "local_csv"

# 模型参数
LIGHTGBM_PARAMS:
  # ... (LGBM 参数)
CNN_PARAMS:
  sequence_length: 60  # CNN 回看窗口长度

# 回测配置 (Backtest Settings) - NEW!
BACKTEST:
  START_DATE: "2025-01-01"   # 回测开始日期 (YYYY-MM-DD)
  END_DATE: "2025-12-31"     # 回测结束日期
  INITIAL_BALANCE: 10000.0   # 初始资金
  TRANSACTION_COST: 0.00005  # 交易成本 (例如 0.5 pips = 0.00005 价格单位)
  ENSEMBLE: true             # 是否启用模型集成 (LGBM + CNN)
```

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

---

## 🏃‍♂️ 详细使用教程

### 第一步: 训练模型 (Train)
运行训练流水线。此步骤将：
1. 下载或加载历史数据。
2. 计算技术指标。
3. **保存 Scaler**: 将训练集拟合的 Scaler 保存到 `models/scaler.pkl` (关键！)。
4. 训练 LightGBM 和 CNN (如果安装了 TensorFlow)。
5. 保存模型到 `models/` 目录。

```bash
python src/train.py
```
*输出：`models/` 文件夹下生成 `lgbm_model.txt`, `cnn_model.keras` 和 `scaler.pkl`。*

### 第二步: 策略回测 (Backtest)
使用**从未见过**的数据进行回测。回测引擎会自动加载第一步保存的模型和 Scaler，严格避免数据泄露。

1. 确保 `config.yaml` 中的 `BACKTEST` 日期范围没有在训练集中使用过（虽然代码不做强制检查，但建议人工划分，例如 2020-2022 训练，2023 测试）。
2. 运行回测：

```bash
# 推荐使用模块方式运行，避免路径报错
python -m src.backtest
```

**回测输出:**
- 控制台打印详细指标：
    - **Net Profit**: 扣除交易成本后的净利润。
    - **ROI**: 投资回报率。
    - **Sharpe Ratio**: 夏普比率 (年化)。
    - **Max Drawdown**: 最大回撤。
    - **Trades**: 交易次数。
- 图片结果：`data/{SYMBOL}_backtest.png` (包含资金曲线对比)。

### 第三步: 启动 API 服务 (Inference)
启动本地预测服务，用于接入交易终端或前端展示。

```bash
uvicorn src.api:app --reload
```
*   服务启动后，访问文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   **获取预测接口**:
    ```http
    GET /predict/EURUSD
    ```
    **响应示例**:
    ```json
    {
      "symbol": "EURUSD",
      "timestamp": "2023-10-27T14:00:00Z",
      "predictions": {
        "dataset": "valid",
        "ensemble_signal": "Bullish",
        "confidence": 0.65
      }
    }
    ```

---

## 📂 项目结构

```
Gemini/2.0/
├── config/             # 配置文件
│   └── config.yaml     # 核心配置 (参数、回测设置)
├── data/               # 存放下载的数据和回测结果图片
├── models/             # 存放训练好的模型 (LightGBM, CNN) 和 Scaler
├── notebooks/          # Jupyter Notebooks (用于 Colab 实验)
├── src/                # 源代码
│   ├── api.py          # FastAPI 接口服务
│   ├── backtest.py     # 优化后的回测引擎
│   ├── data_loader.py  # 数据加载与缓存
│   ├── models.py       # 模型定义 (Model Factory)
│   ├── preprocessing.py# 特征工程与数据预处理
│   ├── train.py        # 训练流水线入口
│   └── __init__.py
├── requirements.txt    # 项目依赖
├── README.md           # 项目文档
└── prompt.txt          # 原始需求描述
```
