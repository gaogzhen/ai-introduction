# 导入必要的库
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


matplotlib.use('TkAgg')

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ======================
# 1. 加载数据
# ======================
# 示例数据集（可以替换为你的 CSV 文件路径）
# 假设数据文件包含特征列 'AT', 'V', 'AP', 'RH' 和目标列 'PE'
data = pd.read_csv('./ccpp.csv')  # 替换为你的文件路径

# 或者使用 scikit-learn 内置数据集（例如波士顿房价）
# from sklearn.datasets import load_boston
# boston = load_boston()
# data = pd.DataFrame(boston.data, columns=boston.feature_names)
# data['PRICE'] = boston.target

# ======================
# 2. 数据预处理
# ======================
# 提取特征 X 和目标 y
X = data[['AT', 'V', 'AP', 'RH']]  # 特征列
y = data['PE']                     # 目标列

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 3. 训练线性回归模型
# ======================
model = LinearRegression()       # 创建模型
model.fit(X_train, y_train)      # 训练模型

# ======================
# 4. 预测与评估
# ======================
y_pred = model.predict(X_test)   # 预测测试集

# 计算误差指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"模型系数: {model.coef_}")    # 特征权重
print(f"截距项: {model.intercept_}") # 截距
print(f"均方误差 (MSE): {mse:.2f}")  # 误差越小越好
print(f"R² 分数: {r2:.2f}")         # 越接近1越好

# ======================
# 5. 可视化结果
# ======================
# 绘制真实值与预测值对比图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 理想对角线
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("真实值 vs 预测值")
plt.show()

# 绘制残差图（误差分布）
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='dashed')
plt.xlabel("预测值")
plt.ylabel("残差")
plt.title("残差分析")
plt.show()