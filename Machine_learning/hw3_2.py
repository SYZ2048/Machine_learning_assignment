#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment
@File    ：hw3_2.py.py
@IDE     ：PyCharm
@Author  ：SYZ
@Date    ：2023/12/7 14:31
布瓜养了一群小鸡，小鸡对温度和湿度比较敏感，每天清晨如果感到不舒服会鸡叫吵醒
睡懒觉的布瓜。布瓜统计了 10 天早上的温度、湿度和鸡叫的情况。假设一年中的温度和湿
度均匀分布，温度分布在 0~30°C，湿度分布在 20~60%，请使用逻辑回归的知识预测布瓜有
多少天早上会被吵醒。
reference:  https://towardsdatascience.com/logistic-regression-in-python-a-helpful-guide-to-how-it-works-6de1ef0a2d2
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

temperature = np.array([20, 15, 4.9, 0, 2.5, 2.5, 0, 3, 1, 4]).reshape(-1, 1)
humidity = np.array([60.2, 51, 20, 29.9, 25.2, 24.9, 31, 23.8, 27.6, 22.3]).reshape(-1, 1)
cockcrow = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])

# 特征数据集，包括温度和湿度
features = np.hstack((temperature, humidity))
# print(features)

# 创建逻辑回归模型
model = LogisticRegression()

# 拟合模型
clf = model.fit(features, cockcrow)

predict_label = model.predict(features)
# print(predict_label)

# 打印斜率和截距
print('Intercept (Beta 0): ', clf.intercept_)
print('Slopes (Beta 1 and Beta 2): ', clf.coef_)

# 生成温度和湿度的组合
Number_total = 10000
temp_rand = np.random.uniform(0, 30, Number_total).reshape(-1, 1)  # 温度从0到30度，随机生成10000个
# print(temp_rand[0:10])
humid_rand = np.random.uniform(20, 60, Number_total).reshape(-1, 1)  # 湿度从20%到60%
# print(humid_rand[0:10])
feature_rand = np.hstack((temp_rand, humid_rand))

# 使用模型预测每种组合下的鸡叫概率
cockcrow_result = model.predict(feature_rand)
# print(cockcrow_result[0:100])
# print(np.count_nonzero(cockcrow_result == 1))
cockcrow_days = np.count_nonzero(cockcrow_result == 1) / Number_total * 365
print("cockcrow_days per year: ", cockcrow_days)
