import numpy as np
import sympy as sp


str_f = '4 * x1 ** 2 + (x2 - 2) ** 2'


def f(x1, x2):
    return 4 * x1 ** 2 + (x2 - 2) ** 2


def norma(v):
    sq_sum = 0
    for xi in v:
        sq_sum += xi ** 2
    return sq_sum ** 0.5


def gradient(func, point):
    x1, x2 = sp.symbols('x1 x2')
    grad_vector = []
    for var in [x1, x2]:
        df = sp.diff(4 * x1 ** 2 + (x2 - 2) ** 2, var)
        d = sp.lambdify((x1, x2), df)
        grad_vector.append(-d(*point))
    return grad_vector


eps = 0.01
x1, x2, a = sp.symbols('x1 x2 a')
x = [2, 2]
count = 0
while norma(gradient(f, x)) >= eps:
    for i in range(len(x)):
        s = [1 if i == count else 0 for i in range(len(x))]
        x_with_step = []
        str_eq = str_f
        for i in range(len(x)):
            elem = str(sp.sympify(eval(f'{x[i]} + a * {s[i]}')))
            x_with_step.append(sp.sympify(eval(elem)))
            str_eq = str_eq.replace(f'x{i + 1}', f'({elem})')
        equation = sp.Eq(sp.sympify(str_eq), 0)
        alpha = sp.solve(equation, a)
        x = np.array(x) + alpha * np.array(s)
        count += 1
        count %= 2

print(f(x[0], x[1]))



