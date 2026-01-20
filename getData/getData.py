import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz

# 1. 初始化 MT5 连接
if not mt5.initialize():
    print(f"MT5 初始化失败, 错误代码: {mt5.last_error()}")
    quit()

print(f"MT5 连接成功。终端版本: {mt5.version()}")

# 2. 配置参数
symbols = ["XAUUSD"]  # 目标品种, e.g. "EURUSD", "GBPUSD"
timeframe = mt5.TIMEFRAME_H1              # 1小时周期

# --- 日期配置 (YYYY-MM-DD) ---
start_date_str = "2018-01-01"
end_date_str = "2020-12-31"   # 如果要获取到最新，可以设置为 None 或当天日期
# ------------------------------

# 设置时区 (UTC 以保持一致性)
timezone = pytz.timezone("Etc/UTC")

# 解析日期函数
def parse_date(date_str, tz):
    if date_str is None:
        return datetime.now(tz)
    # 解析字符串并附加时区
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(tzinfo=tz)

try:
    date_from = parse_date(start_date_str, timezone)
    date_to = parse_date(end_date_str, timezone)
    
    # 如果结束时间早于开始时间
    if date_to < date_from:
        print("错误: 结束日期不能早于开始日期")
        mt5.shutdown()
        quit()
        
except ValueError as e:
    print(f"日期格式错误 (请使用 YYYY-MM-DD): {e}")
    mt5.shutdown()
    quit()

print(f"请求数据时间范围: {date_from.strftime('%Y-%m-%d')} 到 {date_to.strftime('%Y-%m-%d')}")

# 3. 循环获取数据并保存
for symbol in symbols:
    print(f"\n正在获取 {symbol} 数据...")

    # 检查品种是否存在于市场报价中，如果不在则尝试添加
    selected = mt5.symbol_select(symbol, True)
    if not selected:
        print(f"无法在 Market Watch 中找到 {symbol}，尝试手动添加...")
    
    # 获取历史数据
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"获取 {symbol} 数据失败 (可能是该品种不存在或历史数据未下载)。错误: {mt5.last_error()}")
        continue

    # 将数据转换为 Pandas DataFrame
    df = pd.DataFrame(rates)

    # 4. 数据清洗与格式化
    # 将时间戳转换为 datetime 格式
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # 重命名列以符合你的 CSV 要求
    df.rename(columns={
        'time': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume' 
    }, inplace=True)

    # 筛选需要的列
    output_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # 生成文件名 (包含日期范围)
    s_str = date_from.strftime('%Y%m%d')
    e_str = date_to.strftime('%Y%m%d')
    file_name = f"{symbol}_H1_{s_str}_{e_str}.csv"
    
    # 保存到 CSV
    output_df.to_csv(file_name, index=False)
    
    print(f"成功: {symbol} 已保存至 {file_name}，共 {len(df)} 条记录。")
    
    # 打印前两行用于验证格式
    print("数据预览:")
    print(output_df.head(2))

# 5. 关闭连接
mt5.shutdown()
print("\n所有任务已完成，MT5 连接已关闭。")