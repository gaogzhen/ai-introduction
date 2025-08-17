import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings

# 设置环境变量解决KMeans内存泄漏警告
os.environ['OMP_NUM_THREADS'] = '1'

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning,
                        message="KMeans is known to have a memory leak on Windows with MKL")

# 加载数据
data = pd.read_csv('data.csv')
X = data[['V1', 'V2']].values
y_true = data['labels'].values
new_point = np.array([[80, 60]])

# 1. KMeans 算法实现
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

# 标签对齐
label_map_kmeans = {}
for i in range(3):
    mask = (kmeans_labels == i)
    if sum(mask) > 0:
        true_label = np.argmax(np.bincount(y_true[mask]))
        label_map_kmeans[i] = true_label

# 矫正标签
kmeans_labels_corrected = np.array([label_map_kmeans[label] for label in kmeans_labels])
kmeans_pred = label_map_kmeans[kmeans.predict(new_point)[0]]

# 计算准确率
kmeans_accuracy = accuracy_score(y_true, kmeans_labels_corrected)

# 2. KNN 算法实现
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y_true)
knn_pred = knn.predict(new_point)[0]
knn_accuracy = accuracy_score(y_true, knn.predict(X))

# 3. MeanShift 算法实现
ms = MeanShift(bandwidth=20, bin_seeding=True)
ms_labels = ms.fit_predict(X)

# 标签对齐
label_map_ms = {}
unique_labels = np.unique(ms_labels)
for label in unique_labels:
    mask = (ms_labels == label)
    if sum(mask) > 0:
        true_label = np.argmax(np.bincount(y_true[mask]))
        label_map_ms[label] = true_label

# 矫正标签
ms_labels_corrected = np.array([label_map_ms[label] for label in ms_labels])
ms_pred = label_map_ms[ms.predict(new_point)[0]]

# 计算准确率
ms_accuracy = accuracy_score(y_true, ms_labels_corrected)

# 4. 数据可视化
colors = ['#FF5733', '#33FF57', '#3357FF']  # 红, 绿, 蓝
cmap = ListedColormap(colors)

plt.figure(figsize=(18, 12))

# 1. 原始数据分布
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, s=15, alpha=0.7)
plt.scatter(80, 60, c='gold', marker='*', s=300, edgecolor='black')
plt.title('Original Data Distribution')
plt.xlabel('V1')
plt.ylabel('V2')
plt.grid(alpha=0.3)

# 2. KMeans聚类结果
plt.subplot(2, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels_corrected, cmap=cmap, s=15, alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids')
plt.scatter(80, 60, c='gold', marker='*', s=300, edgecolor='black')
plt.title(f'KMeans Clustering (Acc: {kmeans_accuracy:.2%})\nPredicted Class: {kmeans_pred}')
plt.xlabel('V1')
plt.ylabel('V2')
plt.grid(alpha=0.3)
plt.legend()

# 3. KNN分类结果
plt.subplot(2, 2, 3)
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cmap, s=15, edgecolor='k', alpha=0.7)
plt.scatter(80, 60, c='gold', marker='*', s=300, edgecolor='black')
plt.title(f'KNN Classification (Acc: {knn_accuracy:.2%})\nPredicted Class: {knn_pred}')
plt.xlabel('V1')
plt.ylabel('V2')
plt.grid(alpha=0.3)

# 4. MeanShift聚类结果
plt.subplot(2, 2, 4)
plt.scatter(X[:, 0], X[:, 1], c=ms_labels_corrected, cmap=cmap, s=15, alpha=0.7)
plt.scatter(ms.cluster_centers_[:, 0], ms.cluster_centers_[:, 1],
            c='black', marker='X', s=200, label='Centroids')
plt.scatter(80, 60, c='gold', marker='*', s=300, edgecolor='black')
plt.title(f'MeanShift Clustering (Acc: {ms_accuracy:.2%})\nPredicted Class: {ms_pred}')
plt.xlabel('V1')
plt.ylabel('V2')
plt.grid(alpha=0.3)
plt.legend()

plt.suptitle('2D Data Clustering Comparison', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300)
plt.show()

# 打印结果
print("="*50)
print(f"KMeans 结果: 准确率 = {kmeans_accuracy:.2%}, 新点(80,60)预测 = {kmeans_pred}")
print(f"KNN 结果:     准确率 = {knn_accuracy:.2%}, 新点(80,60)预测 = {knn_pred}")
print(f"MeanShift 结果: 准确率 = {ms_accuracy:.2%}, 新点(80,60)预测 = {ms_pred}")
print("="*50)