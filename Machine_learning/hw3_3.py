#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_3.py.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/7 20:32
SVM
"""
from sklearn.svm import SVC
import numpy as np

# 给定的数据点和标签
X = np.array([[1, 1], [2, 2], [0, 0], [-1, 0]])
y = np.array([1, 1, -1, -1])

# 创建SVM分类器实例，使用线性核
clf = SVC(kernel='linear')

# 训练模型
clf.fit(X, y)

# 获取超平面的参数
w = clf.coef_[0]
b = clf.intercept_[0]

# 打印超平面参数和支持向量
print("w: ", w)
print("b: ", b)
print("clf.support_vectors_: ", clf.support_vectors_)


