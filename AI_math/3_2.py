import numpy as np
from matplotlib import pyplot as plt

# 给定的矩阵 A 和向量 b
A = np.array([[3, 2, 2], [2, 2, 1], [2, 1, 3]])
# A = np.array([[3, 2, 1], [2, 3, 1], [1, 1, 2]])
b = np.array([1, 2, 3])

# 计算最优解 x_opt np.linalg.inv: A^-1
x_opt = np.linalg.inv(A) @ (b / 2)

# 初始点 x0
x0 = np.array([0, 0, 0])

# 步长 alpha，容差 tol 和最大迭代次数
alpha = 0.05  # 学习率或迭代步长，这个值可能需要调整以确保迭代的稳定性
tolerance = 1e-6
max_iterations = 50  # 最大迭代次数
f_xk = []
dist_xk_sub_xopt = []


# 目标函数
def f(x, A, b):
    return x.T @ A @ x - b.T @ x


# 目标函数的梯度
def grad_f(x, A, b):
    return 2 * A @ x - b


# 最速下降法（梯度下降法）
def gradient_descent(A, b, x0, alpha, tol, max_iterations):
    x = x0
    for i in range(max_iterations):
        gradient = grad_f(x, A, b)
        if np.linalg.norm(gradient) < tol:
            break
        r_i = b - A @ x
        d_i = -gradient
        alpha_i = (d_i.T @ r_i) / (d_i.T @ A @ d_i)
        x = x + alpha_i * d_i  # 更新 x
        # 计算当前解与最优解的距离
        distance = np.linalg.norm(x - x_opt)

        # 打印迭代次数和距离
        print(f"Iteration {i + 1}: Distance to optimal = {distance:.5f}\t ,f(xk)={f(x, A, b)}, alpha_i = {alpha_i}")
        dist_xk_sub_xopt.append(distance)
        f_xk.append(f(x, A, b))
    return x


# 执行梯度下降法
x_opt = gradient_descent(A, b, x0, alpha, tolerance, max_iterations)
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(f_xk)
ax1.set_ylabel('f(xk)')

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(dist_xk_sub_xopt)
ax2.set_ylabel('|xk-x_opt|')
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域。
plt.show()

print("最优解 x_opt:", x_opt)
