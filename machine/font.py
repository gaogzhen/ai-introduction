import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# 设置支持中文的字体（以简体中文为例）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # ✅ 精确名称
plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题

# 测试绘图
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('测试中文')
plt.show()

# import matplotlib.font_manager
#
# # 检查字体是否被 Matplotlib 识别
# font_list = matplotlib.font_manager.findSystemFonts()
# for f in font_list:
#     print(f)
# sc_fonts = [f for f in font_list if "NotoSansCJK" in f and "SC" in f]
# print("简体中文字体路径:", sc_fonts[0] if sc_fonts else "未找到")