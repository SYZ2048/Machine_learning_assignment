import numpy as np
from matplotlib import pyplot as plt
import math
import sympy
import time

max_iterations = 100  # 最大迭代次数
alpha = 0.05  # 学习率或迭代步长，这个值可能需要调整以确保迭代的稳定性
tolerance = 1e-6
x0 = np.array([1, 1, 1])

f1, f2, f3, sym_x, sym_y, sym_z = sympy.symbols("f1 f2 f3 x y z")
f1 = 6 * sympy.atan(sym_x - 10) - 2 * sympy.exp(-sym_y) - 2 * sympy.exp(-sym_z) + 2 * sym_y + 2 * sym_z - 9
f2 = 2 * sympy.atan(sym_x - 10) - 4 * sympy.exp(-sym_y) - sympy.exp(-sym_z) + 7 * sym_y - 2 * sym_z - 3
f3 = 2 * sympy.atan(sym_x - 10) - sympy.exp(-sym_y) - 3 * sympy.exp(-sym_z) - 1 * sym_y + 5 * sym_z - 3
funcs = sympy.Matrix([f1, f2, f3])
args = sympy.Matrix([sym_x, sym_y, sym_z])
jacobian = funcs.jacobian(args)

# sympy.pprint(jacobian_matrix)

values = {sym_x: 1, sym_y: 2, sym_z: 3}  # 定义 x, y, z 的具体值
print(funcs.subs(values))  # 使用 subs 方法代入具体的值
# jacobian_at_values = jacobian_matrix.subs(values)


# # jacobian_matrix_inv = jacobian_matrix.inv()
# # Matrix([[11*x**2/46 - 110*x/23 + 1111/46, -2*x**2/23 + 40*x/23 - 202/23, -3*x**2/23 + 60*x/23 - 303/23], [(-133*exp(3*y)*exp(z) - 38*exp(3*y) - 70*exp(2*y)*exp(z) - 20*exp(2*y))/(1311*exp(3*y)*exp(z) + 874*exp(3*y) + 1564*exp(2*y)*exp(z) + 897*exp(2*y) + 460*exp(y)*exp(z) + 230*exp(y)), (247*exp(3*y)*exp(z) + 133*exp(3*y) + 130*exp(2*y)*exp(z) + 70*exp(2*y))/(1311*exp(3*y)*exp(z) + 874*exp(3*y) + 1564*exp(2*y)*exp(z) + 897*exp(2*y) + 460*exp(y)*exp(z) + 230*exp(y)), (152*exp(3*y)*exp(z) - 19*exp(3*y) + 80*exp(2*y)*exp(z) - 10*exp(2*y))/(1311*exp(3*y)*exp(z) + 874*exp(3*y) + 1564*exp(2*y)*exp(z) + 897*exp(2*y) + 460*exp(y)*exp(z) + 230*exp(y))], [(-8*exp(y)*exp(z) - 3*exp(z))/(69*exp(y)*exp(z) + 46*exp(y) + 46*exp(z) + 23), (5*exp(y)*exp(z) - exp(z))/(69*exp(y)*exp(z) + 46*exp(y) + 46*exp(z) + 23), (19*exp(y)*exp(z) + 10*exp(z))/(69*exp(y)*exp(z) + 46*exp(y) + 46*exp(z) + 23)]])
# print(jacobian_matrix_inv)

x = sympy.Matrix(x0)

for i in range(max_iterations):
    print(i)
    value_x = {sym_x: x[0], sym_y: x[1], sym_z: x[2]}
    x_s = x - jacobian.subs(value_x).inv() @ sympy.Matrix(funcs.subs(value_x))  # 更新 x
    s = 1
    value_x_s = {sym_x: x_s[0], sym_y: x_s[1], sym_z: x_s[2]}
    while sympy.Matrix.norm(funcs.subs(value_x_s)) >= sympy.Matrix.norm(funcs.subs(value_x)) * (1 - s / 2):
        s = s / 2
        x_s = x - s * jacobian.subs(value_x).inv() @ sympy.Matrix(funcs.subs(value_x))  # 更新 x
        value_x_s = {sym_x: x_s[0], sym_y: x_s[1], sym_z: x_s[2]}
    x = x_s

print(x)

# def f1(x):
#     return 6 * math.atan(x[0] - 10) - 2 * math.exp(-x[1]) - 2 * math.exp(-x[2]) + 2 * x[1] + 2 * x[2] - 9
# def f2(x):
#     return 2 * math.atan(x[0] - 10) - 4 * math.exp(-x[1]) - math.exp(-x[2]) + 7 * x[1] - 2 * x[2] - 3
# def f3(x):
#     return 2 * math.atan(x[0] - 10) - math.exp(-x[1]) - 3 * math.exp(-x[2]) - 1 * x[1] + 5 * x[2] - 3
