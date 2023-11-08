import numpy as np

# 给定的矩阵 A
A = np.array([[3, 2, 0], [2, 2, 0], [2, 1, 0]])
m, n = A.shape
alpha = 0.05  # 学习率或迭代步长
tolerance = 1e-6
max_iterations = 100  # 最大迭代次数
x0 = np.zeros((n, 1))  # 初始 x 是一个 n x 1 的向量

# 定义函数、一阶导数和二阶导数
def f(x):
    term1 = -np.sum(np.log(1 - A @ x))
    term2 = -np.sum(np.log(1 - x ** 2))
    return term1 + term2

def Delta1(A, x):
    grad = np.zeros((n, 1))
    for i in range(m):
        ai = A[i, :].reshape(-1, 1)
        grad += ai / (1 - ai.T @ x)
    for j in range(n):
        grad[j] += 2 * x[j] / (1 - x[j] ** 2)
    return grad

def Delta2(A, x):
    H = np.zeros((n, n))
    for i in range(m):
        ai = A[i, :].reshape(-1, 1)
        outer_product = np.outer(ai, ai)
        H += outer_product / (1 - ai.T @ x) ** 2
    for j in range(n):
        H[j, j] += (2 + 2 * x[j] ** 2) / (1 - x[j] ** 2) ** 2
    return H

# 牛顿法优化过程
x = x0
for i in range(max_iterations):
    delta1 = Delta1(A, x)
    delta2 = Delta2(A, x)
    dnt = -np.linalg.inv(delta2) @ delta1

    # 回溯直线搜索
    t = 1
    while f(x + t * dnt) > f(x) + alpha * t * np.dot(delta1.T, dnt):
        t *= 0.5
    x += t * dnt

    # 判断是否达到容差要求
    if np.linalg.norm(dnt) < tolerance:
        break

print("Optimized x:", x)
