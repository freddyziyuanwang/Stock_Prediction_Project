import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 读取 CSV 文件
data = pd.read_csv("AAPL_data.csv", skiprows=2)

# 打印原始列名
print("原始列名：", data.columns)

# 重命名列（确保与文件结构匹配）
data.columns = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

# 打印重命名后的列名
print("重命名后的列名：", data.columns)

# 将 "Date" 设置为索引并转换为日期格式
data.set_index("Date", inplace=True)
data.index = pd.to_datetime(data.index)

# 打印前几行数据
print("数据预览：", data.head())


# 绘制收盘价
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x=data.index, y="Close")
plt.title("AAPL Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(True)
plt.show()

# 计算50天和200天的移动平均线
data["MA_50"] = data["Close"].rolling(window=50).mean()
data["MA_200"] = data["Close"].rolling(window=200).mean()

# 绘制收盘价和移动平均线
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["Close"], label="Close Price")
plt.plot(data.index, data["MA_50"], label="50-Day MA", linestyle="--")
plt.plot(data.index, data["MA_200"], label="200-Day MA", linestyle="--")
plt.title("AAPL Stock Prices with Moving Averages", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# 将日期转换为整数，作为特征
data['Date_Ordinal'] = data.index.map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)  # 特征：日期
y = data['Close'].values  # 标签：收盘价

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 可视化实际价格和预测价格
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, label="Actual Prices", color="blue", alpha=0.6)
plt.plot(X_test, y_pred, label="Predicted Prices", color="red", linewidth=2)
plt.title("AAPL Stock Price Prediction (Linear Regression)", fontsize=16)
plt.xlabel("Date (Ordinal)", fontsize=12)
plt.ylabel("Price (USD)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 打印模型参数
print(f"Model Coefficient: {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

