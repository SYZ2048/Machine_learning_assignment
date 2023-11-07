import numpy as np
from matplotlib import pyplot as plt
import math
import sympy

max_iterations = 100  # 最大迭代次数

f1, f2, f3, x, y, z = sympy.symbols("f1 f2 f3 x y z")
f1 = 6 * sympy.atan(x - 10) - 2 * sympy.exp(-y) - 2 * sympy.exp(-z) + 2 * y + 2 * z - 9
f2 = 2 * sympy.atan(x - 10) - 4 * sympy.exp(-y) - sympy.exp(-z) + 7 * y - 2 * z - 3
f3 = 2 * sympy.atan(x - 10) - sympy.exp(-y) - 3 * sympy.exp(-z) - 1 * y + 5 * z - 3
funcs = sympy.Matrix([f1, f2, f3])
args = sympy.Matrix([x, y, z])
jacobian_matrix = funcs.jacobian(args)
# print(res)
# 定义 x, y, z 的具体值
values = {x: 1, y: 2, z: 3}
# 使用 subs 方法代入具体的值
jacobian_at_values = jacobian_matrix.subs(values)

# 打印结果
print(jacobian_at_values)

def f1(x):
    return 6 * math.atan(x[0] - 10) - 2 * math.exp(-x[1]) - 2 * math.exp(-x[2]) + 2 * x[1] + 2 * x[2] - 9
def f2(x):
    return 2 * math.atan(x[0] - 10) - 4 * math.exp(-x[1]) - math.exp(-x[2]) + 7 * x[1] - 2 * x[2] - 3
def f3(x):
    return 2 * math.atan(x[0] - 10) - math.exp(-x[1]) - 3 * math.exp(-x[2]) - 1 * x[1] + 5 * x[2] - 3
