非常好！你使用 **1D-CNN 的初衷非常合理**：LightGBM 擅长处理结构化特征（如 RSI、MACD、布林带等），但对**原始价格序列的局部时序模式**（如短期趋势、反转形态、波动聚集）建模能力有限。1D-CNN 正是为此设计——通过卷积核自动提取这些局部模式。

然而，从你的训练日志来看，**CNN 模块目前不仅没有弥补 LGBM 的不足，反而可能引入了噪声甚至数据泄露**，导致 ensemble 效果不如纯 LGBM。

---

## ✅ 优化目标
> **让 1D-CNN 成为一个稳定、互补、有信息增益的“特征提取器”或“弱预测器”，而非直接与 LGBM 平权 ensemble。**

---

## 🔧 具体改善建议（按优先级排序）

### 🚫 1. **首要任务：彻底排查并修复时间序列数据泄露**
这是 CNN 表现异常的最可能原因。

#### ❌ 常见错误：
- 在划分 train/val 之前就用整个数据集生成滑动窗口；
- 验证集窗口的起始时间早于训练集结束时间。

#### ✅ 正确做法（伪代码）：
```python
# Step 1: 按时间全局切分
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
val_df   = df.iloc[split_idx:]

# Step 2: 分别生成序列（确保 val 序列完全在 split_idx 之后）
X_train, y_train = create_sequences(train_df, window=60)  # returns (N, 60, features)
X_val,   y_val   = create_sequences(val_df,   window=60)  # 注意：val_df 不能包含 train_df 的任何未来点！

# Step 3: 只对 train 数据 fit Scaler，val 用 transform
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform_3d(X_train)  # 自定义函数，沿 feature 维度标准化
X_val_scaled   = scaler.transform_3d(X_val)
```

> 🔍 **验证方法**：打印 `train_df.index[-1]` 和 `val_df.index[0]`，确保时间不重叠。

---

### 🔄 2. **改变 CNN 的角色：从“分类器”变为“特征提取器”**
不要让 CNN 直接输出三分类概率，而是**提取高维时序特征，输入给 LGBM**。

#### 架构建议：
```
Input (60, n_features)
    ↓
1D-CNN (Conv1D + ReLU + MaxPooling) × 2~3 层
    ↓
GlobalAveragePooling1D 或 Flatten
    ↓
Dense(32, activation='relu') → **输出 32 维时序特征**
    ↓
保存为 .npy 或直接拼接到 LGBM 的原始特征上
```

#### 训练方式：
- **单独预训练 CNN**：用自监督任务（如预测未来波动率、方向）或有监督任务；
- **冻结 CNN 权重**，将输出特征作为 LGBM 的额外输入；
- **最终只训练 LGBM**，避免两个模型互相干扰。

> ✅ 优势：
> - LGBM 仍主导决策，CNN 仅提供补充信息；
> - 避免 CNN 因过拟合或数据泄露污染 ensemble；
> - 更容易解释：哪些时序模式被 LGBM 利用了？

---

### 📐 3. **优化 CNN 输入：聚焦“原始价格+成交量”，而非技术指标**
LGBM 已经用了技术指标，CNN 应专注 LGBM 不擅长的部分。

#### 推荐输入特征（每根 K 线）：
```python
['open', 'high', 'low', 'close', 'volume']  # 如果 volume 不可用，可省略
```
> ⚠️ **不要输入 RSI/MACD 等派生指标**！这会导致信息冗余，且 CNN 无法从中提取新信息。

#### 标准化：
- 对每个样本（60 根 K 线）做 **局部归一化**（非全局！）：
  ```python
  # 以 close 为基准，计算相对变化
  close_0 = sequence[0, 3]  # 第一根 K 线的 close
  sequence_normalized = sequence / close_0  # 所有价格除以 close_0
  ```
  这样 CNN 学到的是**相对形态**（如“锤子线”、“吞没”），而非绝对价格水平。

---

### 🧪 4. **简化标签 & 任务：先做二分类，再扩展**
三分类中 “Neutral” 定义模糊，易导致模型困惑。

#### 建议：
- **先只预测未来 1H 收益率 > 0 or  ±1 ATR**（过滤噪音）；
- 待 CNN 稳定后再加入 Neutral 类。

---

### ⚙️ 5. **调整 CNN 超参数（轻量化 + 正则化）**
你的当前 CNN 可能过深或过拟合。

#### 推荐架构（Keras 示例）：
```python
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(60, 5)),
    MaxPooling1D(2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    GlobalAveragePooling1D(),  # 避免 Flatten 导致参数爆炸
    Dense(32, activation='relu'),
    Dropout(0.3),              # 关键！防止过拟合
    Dense(1, activation='sigmoid')  # 二分类
])
```
- **减少层数**：2 层 Conv1D 足够捕捉局部模式；
- **使用 GlobalAveragePooling**：比 Flatten 更抗过拟合；
- **加入 Dropout**：至少 0.3；
- **学习率**：从 0.001 开始，配合 ReduceLROnPlateau。

---

### 📊 6. **监控指标：关注验证集稳定性**
- 不要只看 accuracy，重点看：
  - **验证 loss 是否持续下降？**
  - **验证 accuracy 是否平稳（无剧烈波动）？**
  - **训练/验证 gap 是否  55%（二分类）；
2. **将 CNN 输出保存为特征文件**（如 `cnn_features.npy`）；
3. **修改 LGBM 训练脚本，加载这些特征作为额外输入**；
4. **回测时只用 LGBM（含 CNN 特征）**，观察是否提升夏普比率。

如果你需要，我可以为你提供：
- 完整的 `create_sequences` 时间安全版本
- CNN 特征提取器 + LGBM 融合训练脚本
- 相对价格归一化代码

请告诉我你希望优先实现哪一部分！