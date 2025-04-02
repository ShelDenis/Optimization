import numpy as np
import sympy as sp


str_f = '4 * x1 ** 2 + (x2 - 2) ** 2'
H = np.array([[8, 0], [0, 2]])
x0 = np.array([2, 2])


def f(x1, x2):
    return 4 * x1 ** 2 + (x2 - 2) ** 2


def gradient(func, point):
    x1, x2 = sp.symbols('x1 x2')
    grad_vector = []
    for var in [x1, x2]:
        df = sp.diff(4 * x1 ** 2 + (x2 - 2) ** 2, var)
        d = sp.lambdify((x1, x2), df)
        grad_vector.append(-d(*point))
    return grad_vector


def conjugate_gradient(x0, eps=1e-6):
    x = x0
    s = -np.array(gradient(f, x))
    while True:
        alpha = np.dot(s, gradient(f, x)) / np.dot(s, np.dot(H, s))
        x_new = x + alpha * s
        if norma(x_new - x) < eps:
            break
        s = -np.array(gradient(f, x_new)) + np.dot(np.dot(s, H), (np.array(gradient(f, x_new)) - np.array(gradient(f, x)))) / np.dot(np.dot(s, H), s)
        x = x_new
    return x


def norma(v):
    sq_sum = 0
    for xi in v:
        sq_sum += xi ** 2
    return sq_sum ** 0.5


result = conjugate_gradient(x0)
res_f = f(*result)
print(f'x* = {result}\n'
      f'f* = {res_f}')




