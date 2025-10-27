import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 检查版本并打印
print(f"scikit-learn版本: {sklearn.__version__}")
print(f"NumPy版本: {np.__version__}")

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def generate_data(n_samples=100, random_state=42):
    """生成带有噪声的正弦波数据"""
    np.random.seed(random_state)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + np.random.uniform(-0.5, 0.5, n_samples).reshape(-1, 1)
    return X, y


def create_models():
    """创建不同复杂度的模型管道"""
    models = {}

    # 欠拟合模型（线性）
    models['underfitting'] = Pipeline([
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

    # 恰好拟合模型（5次多项式）
    models['fitting'] = Pipeline([
        ('poly', PolynomialFeatures(degree=5, include_bias=False)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

    # 过拟合模型（20次多项式）
    models['overfitting'] = Pipeline([
        ('poly', PolynomialFeatures(degree=20, include_bias=False)),
        ('scaler', StandardScaler()),
        ('linear', LinearRegression())
    ])

    return models


def evaluate_model(model, X_train, X_test, y_train, y_test, X_plot):
    """评估模型并返回结果"""
    # 训练模型
    model.fit(X_train, y_train.ravel())

    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_plot_pred = model.predict(X_plot)

    # 计算误差
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)

    return {
        'predictions': y_plot_pred,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'model': model
    }


def main():
    """主函数"""
    # 1. 生成数据
    X, y = generate_data(n_samples=150)

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 3. 创建用于绘图的密集点
    X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

    # 4. 创建和训练模型
    models = create_models()
    print(models)
    results = {}

    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, X_test,
                                       y_train.ravel(), y_test.ravel(), X_plot)

    # 5. 可视化结果
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['欠拟合（线性回归）', '恰好拟合（5次多项式）', '过拟合（20次多项式）']

    for idx, (name, ax) in enumerate(zip(['underfitting', 'fitting', 'overfitting'], axes)):
        result = results[name]

        # 绘制数据点
        ax.scatter(X_train, y_train, color='yellow', alpha=0.7, s=20, label='训练数据')
        ax.scatter(X_test, y_test, color='red', alpha=0.7, s=20, label='测试数据')

        # 绘制真实函数
        true_y = np.sin(X_plot)
        ax.plot(X_plot, true_y, 'g--', linewidth=2, alpha=0.8, label='真实函数')

        # 绘制拟合曲线
        ax.plot(X_plot, result['predictions'], 'b-', linewidth=2, label='拟合曲线')

        # 设置标题和标签
        ax.set_title(titles[idx], fontsize=14, fontweight='bold')
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend()

        # 显示误差指标
        textstr = f'训练MSE: {result["train_mse"]:.4f}\n测试MSE: {result["test_mse"]:.4f}\n训练RMSE: {result["train_rmse"]:.4f}\n测试RMSE: {result["test_rmse"]:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 6. 打印详细的误差分析
    print("\n" + "=" * 60)
    print("模型性能对比分析")
    print("=" * 60)

    for name in ['underfitting', 'fitting', 'overfitting']:
        result = results[name]
        generalization_gap = result['test_mse'] - result['train_mse']
        print(f"\n{name.upper():<15}")
        print(f"  训练MSE: {result['train_mse']:.6f}")
        print(f"  测试MSE: {result['test_mse']:.6f}")
        print(f"  训练RMSE: {result['train_rmse']:.6f}")
        print(f"  测试RMSE: {result['test_rmse']:.6f}")
        print(f"  泛化差距: {generalization_gap:.6f}")


if __name__ == "__main__":
    main()