import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


matplotlib.use('TkAgg')
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
data = pd.read_csv('examdata.csv')
X = data[['Exam1', 'Exam2']].values  # 关键修改：转为 numpy 数组
# print(X)
y = data['Pass'].values
# print(y)

# 创建模型流水线
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    LogisticRegression(penalty='l2', solver='lbfgs')
)
model.fit(X, y)

# 生成网格点
x_min, x_max = data['Exam1'].min() - 5, data['Exam1'].max() + 5
y_min, y_max = data['Exam2'].min() - 5, data['Exam2'].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# 预测网格点的分类结果
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 可视化
plt.figure(figsize=(10, 6))
# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
# 绘制原始数据点（通过 numpy 布尔索引）
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='red', label='Failed', edgecolors='k')
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='blue', label='Passed', edgecolors='k')
# 标记新样本
plt.scatter(75, 60, c='green', s=200, marker='*',
            label='New Sample (75, 60)')

plt.xlabel('Exam1 Score')
plt.ylabel('Exam2 Score')
plt.title('Logistic Regression with 2nd-Order Boundary')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()