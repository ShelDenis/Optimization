import sympy as sp

x_dct = {
    1: 'x1**2',
    2: 'x2**2',
    3: 'x3**2',
    4: 'x1*x2',
    5: 'x1*x3',
    6: 'x2*x3',
    7: 'x1',
    8: 'x2',
    9: 'x3',
    10: ''
}


def gradient(str_func, point):
    x1, x2, x3 = sp.symbols('x1 x2 x3')
    grad_vector = []
    for var in [x1, x2, x3]:
        func = eval(str_func)
        df = sp.diff(func, var)
        d = sp.lambdify((x1, x2, x3), df)
        grad_vector.append(d(*point))
    return grad_vector


def translate_to_eval(vector, type='inequality'):
    expr = ''
    for i in range(len(vector)):
        if i != 0:
            if vector[i] > 0:
                expr += '+'
            elif vector[i] < 0:
                expr += ''
            else:
                continue
        expr += str(vector[i]) + '*' + x_dct[i + 1]
    if type == 'inequality':
        expr = expr[:-1]
        b = vector[-1]
        expr += f'-{vector[-1]}'
        return expr, b
    else:
        expr = expr[:-1]
        return expr


with open('data/pos_dirs_data.txt') as f:
    data = f.read().splitlines()

L = data[0]
x = [u for u in data[9].split()]
zs = [[u for u in data[i].split()] for i in range(11, 14)]
n_ogr = int(data[1])
ogrs_to_gradient = []
for i in range(n_ogr):
    ogr, b = translate_to_eval([float(x) for x in data[2 + i].split()])
    ogr_copy = ogr[:]
    for i in range(len(x)):
        ogr = ogr.replace(f'x{i + 1}', f'({x[i]})')
    num = sp.sympify(eval(ogr))
    if num == b:
        ogrs_to_gradient.append(ogr_copy)
        print(num)

grad_vector = gradient(ogrs_to_gradient[0], [float(u) for u in x])
print(grad_vector)

for z in zs:
    z = [float(u) for u in z]
    test_val = sum([grad_vector[i] * z[i] for i in range(len(grad_vector))])
    if test_val < 0:
        print(f"Направление ({', '.join([str(u) for u in z])}) является возможным")
    else:
        print(f"Направление ({', '.join([str(u) for u in z])}) не является возможным")

func_grad = gradient(translate_to_eval([float(u) for u in data[0][:-4].split()], type='L'),
                     [float(u) for u in x])
if data[0][-3:] == 'max':
    func_grad = [-u for u in func_grad]

for z in zs:
    z = [float(u) for u in z]
    test_val = sum([func_grad[i] * z[i] for i in range(len(func_grad))])
    if test_val < 0:
        print(f"Направление ({', '.join([str(u) for u in z])}) - спуска")
    else:
        print(f"Направление ({', '.join([str(u) for u in z])}) не является спуском")