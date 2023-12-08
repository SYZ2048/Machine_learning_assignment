#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_6.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/8 20:07
下载sklearn.datasets内置的手写数字数据集，使用前330个样本进行半监督学习测试。具
体来说，用前20个带标签的样本训练一个标签传播模型，对剩余的样本进行分类，将分类效
果最差的5个样本连同其真实标签一起加入原来的训练集中，迭代若干次，观察效果，作业
中对代码步骤简单说明。
"""

from sklearn.datasets import load_digits
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

Iterarion_Max = 10
Data_num = 330
New_add_num = 5

# 下载手写数字数据集
X_digits, y_digits = load_digits(return_X_y=True)  # len(y_digits) = 1797
# X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, random_state=42)
# 使用前330个样本进行半监督学习测试
X = X_digits[:Data_num]
y = y_digits[:Data_num]
# print(y[0:20])

# 创建标签传播模型
label_spread = LabelSpreading(kernel='knn', alpha=0.2)

# 用前20个带标签的样本训练一个标签传播模型，labels代表已知标签
labels = np.full(y.shape, -1)
labels[:20] = y[:20]
# print(labels[0:30])
# print(labels)
for i in range(Iterarion_Max):
    # 训练模型
    label_spread.fit(X, labels)
    print(f"Iteration time {i + 1} accuracy: ", label_spread.score(X, y))

    # 将分类效果最差的5个样本连同其真实标签一起加入原来的训练集中
    predict_labels = label_spread.transduction_  # transduction_:模型对所有数据点（包括最初未标记的那些）的标签预测
    # print("predict_labels: ", predict_labels)
    New_add_num = 5
    for j in range(20, Data_num):
        if y[j] != predict_labels[j]:
            labels[j] = y[j]
            New_add_num -= 1
        if New_add_num == 0:
            break
    # print("labels: ", labels)
