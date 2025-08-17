import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# load data
data = pd.read_csv("examdata.csv")

# visualize data

matplotlib.use('TkAgg')
# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 根据字体名称调整
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# fig1 = plt.figure()
# plt.scatter(data.loc[:, 'Exam1'], data.loc[:, 'Exam2'])
# plt.title('Exam1-Exam2')
# plt.xlabel('Exam1')
# plt.ylabel('Exam2')
# plt.show()

# add label mask
mask = data.loc[:, 'Pass']

fig2 = plt.figure()
passed = plt.scatter(data.loc[:, 'Exam1'][mask], data.loc[:, 'Exam2'][mask])
failed = plt.scatter(data.loc[:, 'Exam1'][~mask], data.loc[:, 'Exam2'][~mask])
plt.title('Exam1-Exam2')
plt.xlabel('Exam1')
plt.ylabel('Exam2')
plt.legend((passed, failed), ('passed', 'failed'))
plt.show()
