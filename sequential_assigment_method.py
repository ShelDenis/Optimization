import sympy as sp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import random

x1, x2 = sp.symbols('x1 x2')


def f1(x):
    return 2 * x[0] ** 2 + 5 * x[1] ** 2


def f2(x):
    return 10 * (x[0] - 5) ** 2 + 3 * (x[1] + 2) ** 2


def f3(x):
    return 7 * (x[0] + 2) ** 2 + (x[1] - 5) ** 2


fs = [2 * x1 ** 2 + 5 * x2 ** 2,
      10 * (x1 - 5) ** 2 + 3 * (x2 + 2) ** 2,
      7 * (x1 + 2) ** 2 + (x2 - 5) ** 2]

func_objs = [f1, f2, f3]

bounds = [
    NonlinearConstraint(f1, -np.inf, 22),
    NonlinearConstraint(f2, -np.inf, 33),
    NonlinearConstraint(f3, -np.inf, 44)
]

n_weights_changes = 10

best_x, best_f, best_weights = [], float('inf'), []

x = np.array([1, 1])

d = 0
cur_bounds = []

for i in range(len(fs)):
    func = sp.lambdify((x1, x2), fs[i] + d, 'numpy')
    result = minimize(func_objs[i], x, method='SLSQP', constraints=cur_bounds)
    d = random.uniform(0, 10)
    bound = NonlinearConstraint(func_objs[i], -np.inf, d * func_objs[i](x))
    cur_bounds.append(bound)

print(f'x* = {result.x}')
print(f'f* = {result.fun}')
