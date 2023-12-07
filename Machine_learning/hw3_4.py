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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X, y = load_boston(True)
# 70%用于训练，30%用于测试
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 线性回归
lr = LinearRegression()
# 使用训练数据进行参数估计
lr.fit(X_train, y_train)

# 输出线性回归的系数
print('线性回归的系数为:\n w = %s \n b = %s' % (lr.coef_, lr.intercept_))