import numpy as np
import math

max_iterations = 100  # 最大迭代次数
alpha = 0.05  # 学习率或迭代步长，这个值可能需要调整以确保迭代的稳定性
tolerance = 1e-6
x0 = np.array([1, 1, 1])


def f(x):
    f1 = 6 * math.atan(x[0] - 10) - 2 * math.exp(-x[1]) - 2 * math.exp(-x[2]) + 2 * x[1] + 2 * x[2] - 9
    f2 = 2 * math.atan(x[0] - 10) - 4 * math.exp(-x[1]) - math.exp(-x[2]) + 7 * x[1] - 2 * x[2] - 3
    f3 = 2 * math.atan(x[0] - 10) - math.exp(-x[1]) - 3 * math.exp(-x[2]) - 1 * x[1] + 5 * x[2] - 3
    return np.array([f1, f2, f3]).flatten()


def jacobian(x):
    result = np.array([[6 / ((x[0] - 10) ** 2 + 1), 2 + 2 * math.exp(-x[1]), 2 + 2 * math.exp(-x[2])],
                       [2 / ((x[0] - 10) ** 2 + 1), 7 + 4 * math.exp(-x[1]), -2 + math.exp(-x[2])],
                       [2 / ((x[0] - 10) ** 2 + 1), -1 + math.exp(-x[1]), 5 + 3 * math.exp(-x[2])]])
    return result


x = x0

for i in range(max_iterations):
    print(i)
    x_s = x - np.linalg.inv(jacobian(x)) @ f(x)  # 更新 x
    # Armijo’s rule
    s = 1
    while np.linalg.norm(f(x_s)) >= np.linalg.norm(f(x)) * (1 - s / 2):
        s = s / 2
        x_s = x - s * np.linalg.inv(jacobian(x)) @ f(x)  # 更新 x
    x = x_s
    if np.linalg.norm(f(x)) < tolerance:
        break

print(x)

