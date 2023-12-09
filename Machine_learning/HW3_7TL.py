#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment
@File    ：hw3_8.py.py
@IDE     ：PyCharm
@Author  ：SYZ
@Date    ：2023/12/8 21:52
下载Fashion MNIST 数据集进行迁移学习测试。假设存在一个模型使用除凉鞋和衬衫之
外的8个类别训练，并获得了超过90%的分类精度，称为模型A。现在你需要处理另一项任务：
你有凉鞋和衬衫的图像，想要训练一个二元分类器。但你的数据集非常小，只有200 张带标
签的图像，试训练得到一个模型B，其架构与模型A 相同，仅微调最后的输出层，在测试集
上查看分类效果。再将之前的层解冻，与输出层一起训练，查看分类效果。
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import math
from functools import partial
import os


# 对输入的图像X和label y进行划分
# 输出值：模型A的X和y（包含8个类别） 模型B的X和y（包含凉鞋和衬衫类别），模型B中，shirts为1，sandals为0
def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6)  # sandals or shirts
    y_A = y[~y_5_or_6]  # y_A为除凉鞋和衬衫之外的8个类别，用于训练模型A
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)?
    return (X[~y_5_or_6], y_A), (X[y_5_or_6], y_B)


# 加载Fashion MNIST 数据集（衣物分类数据集）
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# 像素灰度值归一化
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
# 取5000个数据
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 仅有200个可用的数据集用于数据集B的训练
(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

# 设置随机状态
tf.random.set_seed(42)
np.random.seed(42)

# 训练model A
model_A = keras.models.Sequential()  # 创建一个序贯模型，这是一个线性堆叠的层次结构，每层只有一个输入和一个输出
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))  # 将28x28的图像数据展平为一维数组，作为网络的输入层

# 创建多个全连接层，每个全连接层的神经元数量为（300, 100, 50, 50, 50)，激活函数为selu
for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation="selu"))

# 创建全连接输出层，神经元数量为8，激活函数为softmax，用于多类别（8个类别）的分类
model_A.add(keras.layers.Dense(8, activation="softmax"))

# 编译模型A
model_A.compile(loss="sparse_categorical_crossentropy",     # 损失函数sparse_categorical_crossentropy
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),     # 随机梯度下降
                metrics=["accuracy"])
history = model_A.fit(X_train_A, y_train_A, epochs=20,
                      validation_data=(X_valid_A, y_valid_A))
Loss_model_A, Accuracy_model_A = model_A.evaluate(X_test_A, y_test_A)
print("Loss_model_A: ", Loss_model_A)
print("Accuracy_model_A: ", Accuracy_model_A)
# model_A.save("my_model_A.h5")


# 迁移学习训练model B
# model_A = keras.models.load_model("my_model_A.h5")

model_B_on_A = keras.models.Sequential(model_A.layers[:-1])     # 除了最后一个输出层，其架构与模型A相同

# 创建全连接输出层，神经元数量为1，激活函数为sigmoid，用于二分类
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))

# 复制模型A的权重
model_A_clone = keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())

# 冻结除了最后一个输出层的预训练层的权重
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

# 编译模型B
model_B_on_A.compile(loss="binary_crossentropy",    # 损失函数binary_crossentropy
                     optimizer="sgd",               # 随机梯度下降
                     metrics=["accuracy"])

# 用模型B的200个样本训练4个epochs
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

# 解冻预训练层的权重
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=1e-4)

# # 用模型B的200个样本训练16个epochs，低learning_rate微调模型权重
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))

# 返回损失值（Loss）：这是模型在测试数据上的总损失。对于二元分类问题，通常是二元交叉熵损失。
# 准确率（Accuracy）：这表示模型在测试集上正确分类图像的比例。
Loss_model_B_on_A, Accuracy_model_B_on_A = model_B_on_A.evaluate(X_test_B, y_test_B)
print("Loss_model_B_on_A: ", Loss_model_B_on_A)
print("Accuracy_model_B_on_A: ", Accuracy_model_B_on_A)
