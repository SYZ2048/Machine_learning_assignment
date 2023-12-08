#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_5.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/7 21:29
使用多层感知机的知识，分别训练预测波士顿放假和手写数字数据集。
reference:  https://blog.csdn.net/admin11111111/article/details/116357375
            https://zhuanlan.zhihu.com/p/633131044
"""

from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据集
X_boston, y_boston = load_boston(return_X_y=True)
X_digits, y_digits = load_digits(return_X_y=True)  # len(y_digits) = 1797
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, random_state=42)
X_train_boston, X_test_boston, y_train_boston, y_test_boston = train_test_split(X_boston, y_boston, random_state=42)

# 波士顿房价数据集 - 回归问题
# 数据标准化
scaler_boston = StandardScaler()
X_train_boston_scaled = scaler_boston.fit_transform(X_train_boston)
X_test_boston_scaled = scaler_boston.transform(X_test_boston)

# 创建多层感知机回归模型
mlp_regressor = MLPRegressor(random_state=42, max_iter=500)
mlp_regressor.fit(X_train_boston_scaled, y_train_boston)
# 进行预测
y_pred_boston = mlp_regressor.predict(X_test_boston_scaled)

# 计算准确率
score_boston = mlp_regressor.score(X_test_boston_scaled, y_test_boston)
print("score of boston: ", score_boston)
# 计算均方误差
# mse_boston = mean_squared_error(y_test_boston, y_pred_boston)
# print("mse_boston: ", mse_boston)


# 手写数字数据集 - 分类问题
# 数据标准化
scaler_digits = StandardScaler()
X_train_digits_scaled = scaler_digits.fit_transform(X_train_digits)
X_test_digits_scaled = scaler_digits.transform(X_test_digits)

# 创建多层感知机分类模型
mlp_classifier = MLPClassifier(random_state=42, max_iter=500)
mlp_classifier.fit(X_train_digits_scaled, y_train_digits)
# 进行预测
y_pred_digits = mlp_classifier.predict(X_test_digits_scaled)

# 计算准确率
accuracy_digits = accuracy_score(y_test_digits, y_pred_digits)
print("accuracy_digits: ", accuracy_digits)

