def f(x):
    return x ** 5 - x ** 2


eps = 0.01
a = float(input('a = '))
b = float(input('b = '))
l = b - a
t = 0.618
x1, x2 = a + l * t, b - l * t
f1, f2 = f(x1), f(x2)
while l > eps:
    if f1 > f2:
        b = x1
        f1 = f2
        x1 = x2
        l = b - a
        x2 = b - l * t
        f2 = f(x2)
    else:
        a = x2
        f2 = f1
        x2 = x1
        l = b - a
        x1 = a + l * t
        f1 = f(x1)

print(min(f1, f2))