def f(x):
    return x ** 5 - x ** 2


def get_fib_list(n):
    fib_list = [1, 1]
    for i in range(n - 2):
        fib_list.append(fib_list[-1] + fib_list[-2])
    return fib_list


fib_numbers = get_fib_list(17)
a = float(input('a = '))
b = float(input('b = '))
l_interval = 0.01
near_fib = (b - a) / l_interval
fn = 0
for num in fib_numbers:
    if num > near_fib:
        fn = num
        break
n = fib_numbers.index(fn)


x1 = a + fib_numbers[n - 2] / fib_numbers[n] * (b - a)
x2 = a + fib_numbers[n - 1] / fib_numbers[n] * (b - a)

f1, f2 = f(x1), f(x2)

cur_inl = []
while n > 1:
    n -= 1
    if f1 < f2:
        cur_inl = [a, x2]
        x1 = x2 - fib_numbers[n - 2] / fib_numbers[n] * (cur_inl[1] - cur_inl[0])
    else:
        cur_inl = [x1, b]
        x2 = x1 + fib_numbers[n - 1] / fib_numbers[n] * (cur_inl[1] - cur_inl[0])

print(min(f1, f2))
