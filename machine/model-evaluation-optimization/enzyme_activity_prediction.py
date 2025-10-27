"""
酶活性预测：
    1. 基于T-R-train.csv数据集，建立线性回归模型，计算其在T-R-test.csv数据集上的r2分数，可视化模型预测结果。
    2. 加入多项式特征（2次、5次），建立回归模型。
    3. 计算多项式回归模型对测试数据进行预测的r2分数，判断那个模型预测更准确。
    4. 可视化多项式回归模型的预测结果，判断哪个模型预测更准确。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import seaborn as sns

# 设置图形样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用通用的无衬线字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
train_data = pd.read_csv('..\\..\\data\\T-R-train.csv')
test_data = pd.read_csv('..\\..\\data\\T-R-test.csv')


# 准备数据
X_train = train_data[['T']].values
y_train = train_data['rate'].values
X_test = test_data[['T']].values
y_test = test_data['rate'].values

# 1. 线性回归模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# 预测并计算r2分数
y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

print("=" * 50)
print("ENZYME ACTIVITY PREDICTION - MODEL COMPARISON")
print("=" * 50)
print(f"Linear regression model R2 score: {r2_linear:.4f}")

# 2. 多项式回归模型（2次和5次）
poly_2_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('linear', LinearRegression())
])

poly_5_model = Pipeline([
    ('poly', PolynomialFeatures(degree=5)),
    ('linear', LinearRegression())
])

# 训练多项式模型
poly_2_model.fit(X_train, y_train)
poly_5_model.fit(X_train, y_train)

# 预测并计算r2分数
y_pred_poly_2 = poly_2_model.predict(X_test)
y_pred_poly_5 = poly_5_model.predict(X_test)

r2_poly_2 = r2_score(y_test, y_pred_poly_2)
r2_poly_5 = r2_score(y_test, y_pred_poly_5)

print(f"2nd degree polynomial R2 score: {r2_poly_2:.4f}")
print(f"5th degree polynomial R2 score: {r2_poly_5:.4f}")

# 模型比较
models = ['Linear', 'Poly-2', 'Poly-5']
r2_scores = [r2_linear, r2_poly_2, r2_poly_5]

best_model_idx = np.argmax(r2_scores)
print(f"\nBEST MODEL: {models[best_model_idx]} (R2 = {r2_scores[best_model_idx]:.4f})")

# 创建预测范围
T_range = np.linspace(40, 85, 300).reshape(-1, 1)

# 预测密集点
y_linear_range = linear_model.predict(T_range)
y_poly_2_range = poly_2_model.predict(T_range)
y_poly_5_range = poly_5_model.predict(T_range)

# 简化可视化 - 避免任何可能导致中文显示的问题
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 线性回归
axes[0].scatter(X_train, y_train, color='blue', alpha=0.7, s=50, label='Train')
axes[0].scatter(X_test, y_test, color='red', alpha=0.7, s=50, label='Test')
axes[0].plot(T_range, y_linear_range, color='green', linewidth=2, label='Linear')
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('Activity')
axes[0].set_title(f'Linear Model\nR2 = {r2_linear:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 图2: 2次多项式
axes[1].scatter(X_train, y_train, color='blue', alpha=0.7, s=50, label='Train')
axes[1].scatter(X_test, y_test, color='red', alpha=0.7, s=50, label='Test')
axes[1].plot(T_range, y_poly_2_range, color='orange', linewidth=2, label='Poly-2')
axes[1].set_xlabel('Temperature')
axes[1].set_ylabel('Activity')
axes[1].set_title(f'2nd Degree Polynomial\nR2 = {r2_poly_2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 图3: 5次多项式
axes[2].scatter(X_train, y_train, color='blue', alpha=0.7, s=50, label='Train')
axes[2].scatter(X_test, y_test, color='red', alpha=0.7, s=50, label='Test')
axes[2].plot(T_range, y_poly_5_range, color='purple', linewidth=2, label='Poly-5')
axes[2].set_xlabel('Temperature')
axes[2].set_ylabel('Activity')
axes[2].set_title(f'5th Degree Polynomial\nR2 = {r2_poly_5:.4f}')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 综合比较图
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.7, s=60, label='Training Data')
plt.scatter(X_test, y_test, color='red', alpha=0.7, s=60, marker='s', label='Test Data')

plt.plot(T_range, y_linear_range, color='green', linewidth=2,
         label=f'Linear (R2={r2_linear:.4f})')
plt.plot(T_range, y_poly_2_range, color='orange', linewidth=2,
         label=f'Poly-2 (R2={r2_poly_2:.4f})')
plt.plot(T_range, y_poly_5_range, color='purple', linewidth=2,
         label=f'Poly-5 (R2={r2_poly_5:.4f})')

plt.xlabel('Temperature', fontsize=12)
plt.ylabel('Enzyme Activity', fontsize=12)
plt.title('Model Comparison: Enzyme Activity vs Temperature', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# 修正排名计算
print("\nMODEL PERFORMANCE SUMMARY:")
print("-" * 40)

# 正确计算排名
sorted_indices = np.argsort(r2_scores)[::-1]  # 从高到低排序的索引
ranks = [0] * len(r2_scores)
for rank, idx in enumerate(sorted_indices):
    ranks[idx] = rank + 1

for i, (model, score) in enumerate(zip(models, r2_scores)):
    print(f"{model:8} | R2 = {score:.4f} | Rank: #{ranks[i]}")

# 详细分析
print("\n" + "="*50)
print("DETAILED ANALYSIS")
print("="*50)

if max(r2_scores) < 0:
    print("WARNING: All models show poor performance (R2 < 0)")
    print("This suggests the models perform worse than a simple mean predictor")
elif max(r2_scores) < 0.5:
    print("Models show moderate predictive ability")
else:
    print("Models show good predictive ability")

if r2_poly_5 > r2_poly_2 and r2_poly_5 > r2_linear:
    print("5th degree polynomial performs best")
    print("Higher complexity captures nonlinear patterns")
elif r2_poly_2 > r2_poly_5 and r2_poly_2 > r2_linear:
    print("2nd degree polynomial performs best")
    print("Balances complexity and generalization")
else:
    print("Linear model performs best")
    print("Relationship is approximately linear")

# 计算训练集上的R2分数作为参考
y_train_pred_linear = linear_model.predict(X_train)
y_train_pred_poly_2 = poly_2_model.predict(X_train)
y_train_pred_poly_5 = poly_5_model.predict(X_train)

r2_train_linear = r2_score(y_train, y_train_pred_linear)
r2_train_poly_2 = r2_score(y_train, y_train_pred_poly_2)
r2_train_poly_5 = r2_score(y_train, y_train_pred_poly_5)

print(f"\nTraining set R2 scores for reference:")
print(f"Linear: {r2_train_linear:.4f}")
print(f"Poly-2: {r2_train_poly_2:.4f}")
print(f"Poly-5: {r2_train_poly_5:.4f}")

# 检查过拟合
print(f"\nOverfitting analysis (Train R2 - Test R2):")
print(f"Linear: {r2_train_linear - r2_linear:.4f}")
print(f"Poly-2: {r2_train_poly_2 - r2_poly_2:.4f}")
print(f"Poly-5: {r2_train_poly_5 - r2_poly_5:.4f}")

if (r2_train_poly_5 - r2_poly_5) > 0.2:
    print("WARNING: 5th degree polynomial may be overfitting")

# 显示预测结果对比
print("\n" + "="*50)
print("PREDICTION COMPARISON")
print("="*50)
print("Test Data Points:")
for i in range(len(X_test)):
    print(f"T={X_test[i][0]:.2f}, Actual={y_test[i]:.4f}, "
          f"Linear={y_pred_linear[i]:.4f}, "
          f"Poly2={y_pred_poly_2[i]:.4f}, "
          f"Poly5={y_pred_poly_5[i]:.4f}")