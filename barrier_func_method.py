import sympy as sp
import numpy as np
from scipy.optimize import minimize

x1, x2 = sp.symbols('x1 x2')

eps = 0.01
max_iter = 10
r = 1

f = 2 * x1 ** 2 + 5 * x2 ** 2

gs = [
    x1 + x2 + 2,
    -x1 + 2 * x2 + 1,
    x1 - 5 * x2 + 3
]
# Они все >= 0

penalty = lambda x_vec: r * sum([
    (1 / g).subs({x1: x_vec[0], x2: x_vec[1]})
    for g in gs
])

func = lambda x_vec: float(f.subs({x1: x_vec[0], x2: x_vec[1]}) + penalty(x_vec))

x = np.array([1, 1])

for k in range(max_iter):
    result = minimize(func, x)
    l_opt, x_opt = result.fun, result.x

    penalty_value = penalty(x_opt)
    print(f'Итерация {k}: штраф = {penalty_value}, оптимальные значения = {x_opt}')

    if abs(penalty_value) <= eps:
        break

    r *= 2
    x = x_opt

print(f"x* = {x_opt}")
print(f"f* = {f.subs({x1: x_opt[0], x2: x_opt[1]}):.2f}")