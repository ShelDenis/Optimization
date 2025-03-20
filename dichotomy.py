def f(x):
    return x ** 5 - x ** 2


eps = 0.01
a = float(input('a = '))
b = float(input('b = '))

while b - a > 2 * eps:
    x = (a + b) / 2

    x1, x2 = x - eps / 2, x + eps / 2
    f1, f2 = f(x1), f(x2)
    if f1 > f2:
        a = x1
    else:
        b = x2

print(f((a + b) / 2))
