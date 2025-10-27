import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

matplotlib.use('TkAgg')
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv("./iris_data.csv")

# 设置绘图风格
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(15, 10))

# 1. 特征分布直方图
plt.subplot(2, 3, 1)
sns.histplot(data=df, x='sepal length', hue='target', element="step", kde=True)
plt.title('萼片长度分布')

plt.subplot(2, 3, 2)
sns.histplot(data=df, x='sepal width', hue='target', element="step", kde=True)
plt.title('萼片宽度分布')

plt.subplot(2, 3, 3)
sns.histplot(data=df, x='petal length', hue='target', element="step", kde=True)
plt.title('花瓣长度分布')

plt.subplot(2, 3, 4)
sns.histplot(data=df, x='petal width', hue='target', element="step", kde=True)
plt.title('花瓣宽度分布')

# 2. 特征关系散点图矩阵
plt.subplot(2, 3, 5)
sns.scatterplot(data=df, x='petal length', y='petal width',
                hue='target', style='target', s=80)
plt.title('花瓣长度 vs 花瓣宽度')
plt.legend(loc='upper left')

plt.subplot(2, 3, 6)
sns.scatterplot(data=df, x='sepal length', y='sepal width',
                hue='target', style='target', s=80)
plt.title('萼片长度 vs 萼片宽度')

plt.tight_layout()
plt.suptitle('鸢尾花数据集特征分布', fontsize=16, y=1.02)
plt.show()

# 3. 特征箱线图对比
plt.figure(figsize=(14, 8))

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
for i, feature in enumerate(features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='target', y=feature, data=df)
    plt.title(f'{feature} 箱线图')

plt.tight_layout()
plt.suptitle('不同鸢尾花品种的特征分布对比', fontsize=16, y=1.02)
plt.show()

# 4. 特征相关性热力图
plt.figure(figsize=(10, 8))
corr = df.drop(columns=['label']).corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('特征相关性热力图')
plt.show()

# 5. 特征对组合关系图
sns.pairplot(df, hue='target', diag_kind='kde',
             plot_kws={'s': 40, 'alpha': 0.8},
             corner=True)
plt.suptitle('特征对组合关系', fontsize=16, y=1.02)
plt.show()

# 6. 决策树模型训练和可视化
X = df.iloc[:, :4]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)

# 决策树可视化
plt.figure(figsize=(24, 12))
plot_tree(dt_model,
          filled=True,
          rounded=True,
          feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
          class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
          proportion=True,
          fontsize=12,
          impurity=False,
          node_ids=True,
          max_depth=4)

plt.title('鸢尾花分类决策树', fontsize=16)
plt.show()

# 7. 特征重要性可视化
feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns)
sorted_importances = feature_importances.sort_values()

plt.figure(figsize=(10, 6))
sorted_importances.plot(kind='barh', color='skyblue')
plt.title('特征重要性排序')
plt.xlabel('重要性分数')
plt.ylabel('特征名称')
for i, v in enumerate(sorted_importances):
    plt.text(v, i, f" {v:.3f}", va='center', fontsize=10)
plt.tight_layout()
plt.show()

# 8. 花瓣特征决策边界可视化（使用花瓣长度和宽度）
plt.figure(figsize=(12, 8))

# 提取花瓣特征
X_petal = df[['petal length', 'petal width']]
y_petal = df['label']

# 训练简化模型
dt_petal = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_petal.fit(X_petal, y_petal)

# 创建网格数据
x_min, x_max = X_petal['petal length'].min() - 0.5, X_petal['petal length'].max() + 0.5
y_min, y_max = X_petal['petal width'].min() - 0.5, X_petal['petal width'].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 预测网格点
Z = dt_petal.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)

# 绘制数据点
sns.scatterplot(x='petal length', y='petal width', hue='target',
                data=df, palette='viridis', s=80,
                edgecolor='black', legend=True)

plt.title('基于花瓣特征的决策边界')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend(title='鸢尾花品种')
plt.show()