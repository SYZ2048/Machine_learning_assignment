#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_1.py.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/7 14:11
布瓜同学学习机器学习课程，前三次小测的成绩分别为 43.5，51.3，56.2。课程要求小测
至少有一次及格才能不挂科，本学期还剩下最后一次小测，请使用最小二乘法相关的知识预
测布瓜同学本学期是否会挂科
reference:  https://zhuanlan.zhihu.com/p/629890871
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来设置字体样式以正常显示中文标签（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 正确输出负数

# 成绩数据
X = np.array([1, 2, 3]).reshape(-1, 1)
Y = np.array([43.5, 51.3, 56.2])

# 创建线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, Y)

# 预测第四次小测的成绩
predicted_score = model.predict([[4]])[0]
print('系数:', model.coef_[0])
print('截距:', model.intercept_)
print("predicted_score: ", predicted_score)

# 绘制拟合曲线和数据点
x_line = np.linspace(1, 4, 100).reshape(-1, 1)
y_line = model.predict(x_line)

plt.scatter(X, Y, color='blue', label='实际成绩')
plt.plot(x_line, y_line, color='red', label='拟合曲线')
plt.scatter(4, predicted_score, color='green', label='预测成绩')
plt.xlabel('小测次数')
plt.ylabel('成绩')
plt.title('成绩趋势预测')
plt.legend()
plt.show()

predicted_score

