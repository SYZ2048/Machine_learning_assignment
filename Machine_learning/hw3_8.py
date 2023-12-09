#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Machine_learning_assignment 
@File    ：hw3_8.py.py
@IDE     ：PyCharm 
@Author  ：SYZ
@Date    ：2023/12/8 21:52
基于mnist手写数字数据集，使用深度学习的方法尝试分类训练与预测。提取代码：
(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential

# 定义不同的神经元数量配置，
neuron_numbers = [32, 64, 128, 256, 512]
# 定义不同的训练集和测试集划分比例
test_sizes = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# 像素灰度值归一化
X_train = X_train / 255.0
X_test = X_test / 255.0

accuracies_test_sizes = []
accuracies_neuron_num = []
accuracies_datasets = []


def train_model(data_train, label_train, test_size=0.25, neuron=64):
    # 划分训练集和测试集
    x_train_split, x_test_split, y_train_split, y_test_split = train_test_split(
        data_train, label_train, test_size=test_size, random_state=42)

    # 训练model A
    model_mnist = keras.models.Sequential()  # 创建一个序贯模型，这是一个线性堆叠的层次结构，每层只有一个输入和一个输出
    model_mnist.add(keras.layers.Flatten(input_shape=[28, 28]))  # 将28x28的图像数据展平为一维数组，作为网络的输入层

    # 创建多个全连接层，每个全连接层的神经元数量为（300, 100, 50, 50, 50)，激活函数为selu
    # for n_hidden in (300, 100, 50, 50, 50):
    #     model_mnist.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_mnist.add(keras.layers.Dense(neuron, activation="selu"))

    # 创建全连接输出层，神经元数量为8，激活函数为softmax，用于多类别（8个类别）的分类
    model_mnist.add(keras.layers.Dense(10, activation="softmax"))

    # 编译模型A
    model_mnist.compile(loss="sparse_categorical_crossentropy",  # 损失函数sparse_categorical_crossentropy
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),  # 随机梯度下降
                        metrics=["accuracy"])
    # history = model_mnist.fit(X_train, y_train, epochs=10, verbose=0)  # verbose设置为0会隐藏所有的训练过程中的输出，这包括每个epoch的进度条和其他信息
    history = model_mnist.fit(x_train_split, y_train_split, epochs=10, verbose=0)

    Loss_model_mnist, Accuracy_model_mnist = model_mnist.evaluate(x_test_split, y_test_split)
    print("*" * 10 + f"\ttest_size = {test_size}\t" + "*" * 10)
    print("Loss_model_mnist: ", Loss_model_mnist)
    print("Accuracy_model_mnist: ", Accuracy_model_mnist)
    return Accuracy_model_mnist


def diff_test_size():
    for size in test_sizes:
        acc = train_model(X_train, y_train, test_size=size)
        accuracies_test_sizes.append(acc)
    # 绘制准确率曲线
    # plt.subplot(1, 1, 1)
    # plt.plot(test_sizes, accuracies_test_sizes, 'o-', color='blue')
    # plt.title('Accuracy vs Test Size')
    # plt.xlabel('Test Size')
    # plt.ylabel('Accuracy')
    # plt.tight_layout()
    # plt.show()


def diff_neuron():
    for neuron_num in neuron_numbers:
        acc = train_model(X_train, y_train, neuron=neuron_num)
        accuracies_neuron_num.append(acc)
    # 绘制准确率曲线
    plt.subplot(1, 1, 1)
    plt.plot(neuron_numbers, accuracies_neuron_num, 'o-', color='blue')
    plt.title('Accuracy vs Neuron Numbers')
    plt.xlabel('Neuron Numbers')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


def train_model_diff_data(data_train, label_train, changed_data_train, changed_label_train, test_size=0.25, neuron=64):
    # 划分训练集和测试集
    x_train_split, _, y_train_split, _ = train_test_split(
        changed_data_train, changed_label_train, test_size=test_size, random_state=42)
    _, x_test_split, _, y_test_split = train_test_split(
        data_train, label_train, test_size=test_size, random_state=42)

    # 训练model A
    model_mnist = keras.models.Sequential()  # 创建一个序贯模型，这是一个线性堆叠的层次结构，每层只有一个输入和一个输出
    model_mnist.add(keras.layers.Flatten(input_shape=[28, 28]))  # 将28x28的图像数据展平为一维数组，作为网络的输入层

    # 创建多个全连接层，每个全连接层的神经元数量为（300, 100, 50, 50, 50)，激活函数为selu
    # for n_hidden in (300, 100, 50, 50, 50):
    #     model_mnist.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_mnist.add(keras.layers.Dense(neuron, activation="selu"))

    # 创建全连接输出层，神经元数量为8，激活函数为softmax，用于多类别（8个类别）的分类
    model_mnist.add(keras.layers.Dense(10, activation="softmax"))

    # 编译模型A
    model_mnist.compile(loss="sparse_categorical_crossentropy",  # 损失函数sparse_categorical_crossentropy
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),  # 随机梯度下降
                        metrics=["accuracy"])
    # history = model_mnist.fit(X_train, y_train, epochs=10, verbose=0)  # verbose设置为0会隐藏所有的训练过程中的输出，这包括每个epoch的进度条和其他信息
    history = model_mnist.fit(x_train_split, y_train_split, epochs=10, verbose=0)

    Loss_model_mnist, Accuracy_model_mnist = model_mnist.evaluate(x_test_split, y_test_split)
    print("*" * 10 + f"\ttest_size = {test_size}\t" + "*" * 10)
    print("Loss_model_mnist: ", Loss_model_mnist)
    print("Accuracy_model_mnist: ", Accuracy_model_mnist)
    return Accuracy_model_mnist


def diff_datasets():
    # 原始数据集
    # {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
    # 创建不平衡的训练数据集
    indices_class_1 = np.where(y_train == 1)[0][:1000]  # 类别1的1000个样本
    indices_class_1_imbalance = np.where(y_train == 1)[0][:1990]  # 类别1的1990个样本
    indices_class_2 = np.where(y_train == 2)[0][:1000]  # 类别2的1000个样本
    indices_class_2_imbalance = np.where(y_train == 2)[0][:10]  # 类别2的10个样本
    indices_class_3 = np.where(y_train == 3)[0][:1000]  # 类别3的1000个样本
    indices_class_4 = np.where(y_train == 4)[0][:1000]  # 类别4的1000个样本
    indices_class_5 = np.where(y_train == 5)[0][:1000]  # 类别5的1000个样本
    indices_class_6 = np.where(y_train == 6)[0][:1000]  # 类别6的1000个样本
    indices_class_7 = np.where(y_train == 7)[0][:1000]  # 类别7的1000个样本
    indices_class_8 = np.where(y_train == 8)[0][:1000]  # 类别8的1000个样本
    indices_class_9 = np.where(y_train == 9)[0][:1000]  # 类别9的1000个样本
    indices_class_0 = np.where(y_train == 0)[0][:1000]  # 类别0的1000个样本
    # # 获取其他类别的索引
    # indices_other_classes = \
    #     np.where((y_train != 1) & (y_train != 2))[0]
    # 合并索引
    balanced_indices = np.concatenate(
        [indices_class_1, indices_class_2, indices_class_3, indices_class_4, indices_class_5, indices_class_6,
         indices_class_7, indices_class_8, indices_class_9, indices_class_0])
    imbalanced_indices = np.concatenate(
        [indices_class_1_imbalance, indices_class_2_imbalance, indices_class_3, indices_class_4, indices_class_5, indices_class_6,
         indices_class_7, indices_class_8, indices_class_9, indices_class_0])

    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    X_train_imbalanced = X_train[imbalanced_indices]
    y_train_imbalanced = y_train[imbalanced_indices]

    for size in test_sizes:
        acc = train_model_diff_data(X_train_balanced, y_train_balanced, X_train_imbalanced, y_train_imbalanced, test_size=size)
        accuracies_datasets.append(acc)
    # 绘制准确率曲线
    plt.subplot(1, 1, 1)
    plt.plot(test_sizes, accuracies_test_sizes, 'o-', color='blue', label="origin dataset")
    plt.plot(test_sizes, accuracies_datasets, 'o-', color='green', label="changed dataset")
    plt.title('Accuracy vs Test Size')
    plt.xlabel('Test Size')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()


def diff_model():
    # 创建全连接模型
    model_dense = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 编译模型
    model_dense.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    # 训练模型
    model_dense.fit(X_train, y_train, epochs=5)

    # 评估模型
    test_loss, test_accuracy = model_dense.evaluate(X_test, y_test)
    print(f"Dense Network - Loss: {test_loss}, Accuracy: {test_accuracy}")

    # 创建CNN模型
    model_cnn = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # 编译模型
    model_cnn.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # 训练模型
    # 注意：需要增加一个维度来表示通道数
    model_cnn.fit(X_train[..., np.newaxis], y_train, epochs=5)

    # 评估模型
    test_loss, test_accuracy = model_cnn.evaluate(X_test[..., np.newaxis], y_test)
    print(f"CNN - Loss: {test_loss}, Accuracy: {test_accuracy}")


if __name__ == '__main__':
    diff_test_size()
    diff_neuron()
    diff_datasets()
    diff_model()
