import numpy as np

np.random.seed(10)
m, n = 60, 80
# A = np.random.randn(m, n)
A = np.array([[3, 2, 0], [2, 2, 0], [2, 1, 0]])
m, n = A.shape
b = np.random.randn(m)

# 迭代的参数
ITER_para = 1000; Tol = 1e-8; lR = 1e-3
# 回溯直线搜索的参数
gamma = 0.5; alpha = 0.05;
# 初始值
x = np.zeros((n,))

#目标函数f(x)
f_x_ = lambda x: - np.sum(np.log(1 - A @ x)) - np.sum(np.log(1 - np.square(x)))
#目标函数一阶导
grad = lambda x: np.sum(A / (1 - A @ x)[:, None], axis=0) + 2 * x / (1 - np.square(x))

#目标函数二阶导
def d2fx(x: np.ndarray) -> np.ndarray:
    grad2 = A.T @ np.diag(1 / np.square(1 - A @ x)) @ A
    grad2 += np.diag((2*np.square(x)+2) / np.square(1-np.square(x)))
    return grad2

# 一些监督训练集的变量
x = np.zeros((n, ))
xprev = x; logger = []; x_path = []
for iter in range(ITER_para):
    dfdx = grad(x)
    move = - np.linalg.solve(d2fx(x), dfdx)
# 令t=1
    t = 1
    while f_x_(x + t * move) > f_x_(x) + alpha * t * dfdx @ move:
        t = t * gamma

    x = x + t * move
    if np.linalg.norm(xprev - x) < Tol: break
    xprev = x

result = f_x_(x)
print(x)
print(result)