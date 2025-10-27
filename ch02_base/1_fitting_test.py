import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

"""
1.生成数据
2.划分训练集和测试集
3.定义模型
4.训练模型
5.预测结果，计算误差
6.可视化结果
"""

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 生成数据

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X) + np.random.uniform(-0.5, 0.5, 100).reshape(-1, 1)

# 画出散点图（3个子图）
fig, ax = plt.subplots(1, 3 , figsize=(15, 4))
ax[0].scatter(X, y, color='y')
ax[0].set_title('欠拟合（直线）')
# ax[0].legend()
ax[1].scatter(X, y, color='y', label='Data Points')
ax[1].set_title('恰好拟合（5次多项式）')
# ax[1].legend()
ax[2].scatter(X, y, color='y', label='Data Points')
ax[2].set_title('过拟合（20次多项式）')
# ax[2].legend()
# plt.show()
# 2. 划分训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. 定义模型
model = LinearRegression()
# 3.1 欠拟合（直线）
x_train1 = X_train
x_test1 = X_test
# 4.1 训练模型
model.fit(x_train1, y_train)
# 5.1 预测结果，计算误差
y_train_pred = model.predict(x_train1)
y_test_pred = model.predict(x_test1)
train_mse1 = mean_squared_error(y_train, y_train_pred)
test_mse1 = mean_squared_error(y_test, y_test_pred)

# 6.1 可视化结果
ax[0].plot(X, model.predict(X), color='b')
ax[0].text(-3, 1, f"训练误差: {train_mse1:.4f}")
ax[0].text(-3, 1.3, f"测试误差: {test_mse1:.4f}")
# plt.show()

# 3.2 恰好拟合（5次多项式）
poly5 = PolynomialFeatures(degree=5)
x_train5 = poly5.fit_transform(X_train)
x_test5 = poly5.transform(X_test)
# 4.2 训练模型
model.fit(x_train5, y_train)
# 5.2 预测结果，计算误差
y_train_pred5 = model.predict(x_train5)
y_test_pred5 = model.predict(x_test5)
train_mse5 = mean_squared_error(y_train, y_train_pred5)
test_mse5 = mean_squared_error(y_test, y_test_pred5)

# 6.2 可视化结果
ax[1].plot(X, model.predict(poly5.fit_transform(X)), color='b')
ax[1].text(-3, 1, f"训练误差: {train_mse5:.4f}")
ax[1].text(-3, 1.3, f"测试误差: {test_mse5:.4f}")
# plt.show()

# 3.3 过拟合（20次多项式）
poly20 = PolynomialFeatures(degree=20)
x_train20 = poly20.fit_transform(X_train)
x_test20 = poly20.transform(X_test)
# 4.3 训练模型
model.fit(x_train20, y_train)
# 5.3 预测结果，计算误差
y_train_pred20 = model.predict(x_train20)
y_test_pred20 = model.predict(x_test20)
train_mse20 = mean_squared_error(y_train, y_train_pred20)
test_mse20 = mean_squared_error(y_test, y_test_pred20)

# 6.3 可视化结果
ax[2].plot(X, model.predict(poly20.fit_transform(X)), color='b')
ax[2].text(-3, 1, f"训练误差: {train_mse20=:.4f}")
ax[2].text(-3, 1.3, f"测试误差: {test_mse20:.4f}")
plt.show()