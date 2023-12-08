#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_4.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/7 20:50
使用线性回归和逻辑回归的知识，分别对波士顿放假的公开数据集进行模型的训练和预测。
reference:  https://blog.csdn.net/admin11111111/article/details/116357375
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

X, y = load_boston(True)
# 70%用于训练，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 可视化目标变量（房价）的分布
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, bins=30)
plt.xlabel('House Prices')
plt.title('Boston House Prices')
plt.show()

# 线性回归
linear_regression = LinearRegression()
# 使用训练数据进行参数估计
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test)
# 输出线性回归的系数
print('线性回归的系数为:\n w = %s \n b = %s' % (linear_regression.coef_, linear_regression.intercept_))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))


# 逻辑回归
# 根据中位数划分为二分类
median_value = np.median(y)
y_class = (y > median_value).astype(int)
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.3, random_state=10)

# 创建和训练模型
logistic_regression = LogisticRegression(max_iter=10000)
logistic_regression.fit(X_train, y_train_class)

# 预测和评估
y_pred_class = logistic_regression.predict(X_test)
print('逻辑回归的系数为:\n w = %s \n b = %s' % (logistic_regression.coef_, logistic_regression.intercept_))
print("Logistic Regression Accuracy:", accuracy_score(y_test_class, y_pred_class))

