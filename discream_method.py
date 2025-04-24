import sympy as sp
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import random

x1, x2 = sp.symbols('x x1 x2')


def f1(x):
    return 2 * x[0] ** 2 + 5 * x[1] ** 2


def f2(x):
    return 10 * (x[0] - 5) ** 2 + 3 * (x[1] + 2) ** 2


def f3(x):
    return 7 * (x[0] + 2) ** 2 + (x[1] - 5) ** 2


fs = [2 * x1 ** 2 + 5 * x2 ** 2,
      10 * (x1 - 5) ** 2 + 3 * (x2 + 2) ** 2,
      7 * (x1 + 2) ** 2 + (x2 - 5) ** 2]

bounds = [
    NonlinearConstraint(f1, -np.inf, 22),
    NonlinearConstraint(f2, -np.inf, 33),
    NonlinearConstraint(f3, -np.inf, 44)
]

n_weights_changes = 10

best_x, best_f, best_weights = [], float('inf'), []

for k in range(n_weights_changes):
    crit_f = 0
    sum_weight = 1
    weights = []
    for f in fs:
        if len(weights) == len(fs) - 1:
            w = 1 - sum(weights)
        else:
            w = random.uniform(0, sum_weight)
        weights.append(w)
        crit_f += w * f
        sum_weight -= w

    func = sp.lambdify((x1, x2), crit_f, 'numpy')
    x0 = np.array([random.uniform(-10, 10), random.uniform(-10, 10)])

    result = minimize(lambda args: func(*args), x0, method='SLSQP', constraints=bounds)

    print(f'Итерация - {k}')
    print(f'Веса: {weights}')
    print(f'x* = {result.x}')
    print(f'f* = {result.fun}')

    if result.fun < best_f:
        best_x = result.x
        best_f = result.fun
        best_weights = weights[:]

print(f"\nЛучшее решение:")
print(f'Веса: {best_weights}')
print(f'x* = {best_x}')
print(f'f* = {best_f}')