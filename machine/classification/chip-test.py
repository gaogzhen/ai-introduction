import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, confusion_matrix

matplotlib.use('TkAgg')
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = pd.read_csv('chip_test.csv')
X = data[['test1', 'test2']].values
y = data['pass'].values

# 生成二阶多项式特征 (包含常数项、线性项和二次项)
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)

# 训练逻辑回归模型（无正则化）- 使用 penalty=None
model = LogisticRegression(penalty=None, max_iter=1000, solver='lbfgs')
model.fit(X_poly, y)

# 模型评估
y_pred = model.predict(X_poly)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

print(f"模型准确率: {accuracy:.4f}")
print("混淆矩阵:")
print(conf_matrix)


# 边界曲线函数
def decision_boundary(x1, x2):
    # 创建测试点并转换为多项式特征
    point = np.array([[x1, x2]])
    point_poly = poly.transform(point)

    # 计算预测概率并返回决策值
    proba = model.predict_proba(point_poly)[0][1]
    return proba - 0.5


# 可视化完整边界曲线
# 创建网格点
x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
x2_min, x2_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300),
                       np.linspace(x2_min, x2_max, 300))

# 计算每个网格点的决策值 - 向量化计算提高效率
points = np.c_[xx1.ravel(), xx2.ravel()]
points_poly = poly.transform(points)
Z = model.predict_proba(points_poly)[:, 1] - 0.5
Z = Z.reshape(xx1.shape)

# 设置图形
plt.figure(figsize=(10, 8))

# 绘制原始数据点
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c='royalblue', marker='o', s=60,
            label='通过 (1)', edgecolor='k')
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c='crimson', marker='x', s=60,
            label='失败 (0)')

# 绘制决策边界 (Z=0的等高线)
contour = plt.contour(xx1, xx2, Z, levels=[0],
                      colors='darkgreen', linewidths=2.5)
plt.clabel(contour, inline=True, fontsize=12)

# 填充决策区域
plt.contourf(xx1, xx2, Z, levels=[-10, 0, 10],
             colors=['crimson', 'royalblue'], alpha=0.15)

# 添加标签和标题
plt.xlabel('测试1', fontsize=12)
plt.ylabel('测试2', fontsize=12)
plt.title('芯片测试结果与决策边界 (二阶多项式)', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(alpha=0.3)

# 添加准确率信息
plt.text(0.75, -0.9, f'准确率: {accuracy:.2%}',
         fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()