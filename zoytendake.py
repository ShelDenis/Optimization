# import sympy as sp
# import numpy as np
#
#
# x_dct = {
#     1: 'x1**2',
#     2: 'x2**2',
#     3: 'x3**2',
#     4: 'x1*x2',
#     5: 'x1*x3',
#     6: 'x2*x3',
#     7: 'x1',
#     8: 'x2',
#     9: 'x3',
#     10: ''
# }
#
#
# def gradient(str_func, point):
#     x1, x2, x3 = sp.symbols('x1 x2 x3')
#     grad_vector = []
#     for var in [x1, x2, x3]:
#         func = eval(str_func)
#         df = sp.diff(func, var)
#         d = sp.lambdify((x1, x2, x3), df)
#         grad_vector.append(d(*point))
#     return grad_vector
#
#
# def func_gradient(func_str, vars):
#     function = sp.sympify(func_str)
#     variables = sp.symbols(vars)
#
#     partial_derivatives = [sp.diff(function, var) for var in variables]
#
#     return partial_derivatives
#
#
# def translate_to_eval(vector, type='inequality'):
#     expr = ''
#     for i in range(len(vector)):
#         if i != 0:
#             if vector[i] > 0:
#                 expr += '+'
#             elif vector[i] < 0:
#                 expr += ''
#             else:
#                 continue
#         expr += str(vector[i]) + '*' + x_dct[i + 1]
#     if type == 'inequality':
#         expr = expr[:-1]
#         b = vector[-1]
#         expr += f'-{vector[-1]}'
#         return expr, b
#     else:
#         expr = expr[:-1]
#         return expr
#
#
# with open('data/pos_dirs_data.txt', 'rt', encoding='utf-8') as f:
#     data = f.read().splitlines()
#
# L_str = data[0]
#
# # проверка на допустимость
# signs = []
# x = [0.5, 0.5, 0.5]
# n_ogr = int(data[1])
# ogrs_to_gradient = []
# is_accept = True
# for i in range(n_ogr):
#     sign = data[2 + i][-2:]
#     signs.append(sign)
#     data[2 + i] = data[2 + i][:-2]
#     ogr, b = translate_to_eval([float(x) for x in data[2 + i].split()])
#     ogr_copy = ogr[:]
#     for i in range(len(x)):
#         ogr = ogr.replace(f'x{i + 1}', f'({x[i]})')
#     num = sp.sympify(eval(ogr))
#     if sign == '<=' and num <= b:
#         ogrs_to_gradient.append(ogr_copy)
#         print(num)
#     elif sign == '=>' and num >= b:
#         ogrs_to_gradient.append(ogr_copy)
#         print(num)
#     else:
#         is_accept = False
#
# if not is_accept:
#     print('Вектор не допустим!')
# else:
#     print('Ок!')
#
# # вектор градиент и новая точка
# s = gradient(translate_to_eval([float(u) for u in data[0][:-4].split()], type='L'),
#                      [float(u) for u in x])
#
# x_with_step = []
# str_eq = L_str
# x1, x2, x3, a = sp.symbols('x1 x2 x3 a')
# for i in range(len(x)):
#     elem = str(sp.sympify(eval(f'{x[i]} + a * {s[i]}')))
#     x_with_step.append(str(sp.sympify(eval(elem))))
#     str_eq = str_eq.replace(f'x{i + 1}', f'({elem})')
#
# # интервал допустимых значений для параметра a0
# ineq_system = []
# for i in range(int(data[1])):
#     ineq = data[2 + i]
#     ineq_s, b = translate_to_eval([float(u) for u in ineq.split()])
#     for j in range(len(x_with_step)):
#         ineq_s = ineq_s.replace(f'x{j + 1}', x_with_step[j])
#
#     # inequation = sp.Eq(sp.sympify(ineq_s), b)
#     rel_inq = ''
#     if signs[i] == '<=':
#         rel_inq = sp.LessThan(eval(ineq_s), b)
#     elif signs[i] == '>=':
#         rel_inq = sp.GreaterThan(eval(ineq_s), b)
#     ineq_system.append(rel_inq)
#
# for i in range(len(x_with_step)):
#     rel_inq = sp.GreaterThan(eval(str(x_with_step[i])), 0)
#     ineq_system.append(rel_inq)
#
# dopust_inl = sp.reduce_inequalities(ineq_system)
#
# # Найдем такую величину шага a0, которая обеспечит функции F максимум.
# lf = translate_to_eval([float(u) for u in L_str.split()[:-1]], 'L')
# str_s = func_gradient(lf, 'x1, x2, x3')
# gr_f_x1_a = ';'.join(str(st) for st in str_s)
# for i in range(len(x_with_step)):
#     gr_f_x1_a = gr_f_x1_a.replace(f'x{i + 1}', x_with_step[i])
# gr_f_x1_a_list = gr_f_x1_a.split(';')
# gr_f_x1_a = []
# for exp in gr_f_x1_a_list:
#     gr_f_x1_a.append(exp)
#
# grad_mul = []
# for i in range(len(x)):
#     grad_mul.append(f'({gr_f_x1_a[i]}) * {str(x[i])}')
# grad_mul = '+'.join(grad_mul)
#
# equation = sp.Eq(eval(grad_mul), 0)
# a_opt = sp.solve(equation, 'a')[0]
#
# expr = str(dopust_inl).replace('a', 'a_opt')
# expr = expr.replace('-oo', 'float("-inf")')
# expr = expr.replace('oo', 'float("inf")')
# # если альфа в интервале, то оставляем ее как есть
# ars = dopust_inl.args[0].args[0]
# left_bound, right_bound = (float(str(dopust_inl.args[0].args[0]).split()[0]),
#                            float(str(dopust_inl.args[-1].args[0]).split()[-1]))
#
# # находим ближайшую границу
# if abs(a_opt - right_bound) < abs(a_opt - left_bound):
#     a_opt = right_bound
# else:
#     a_opt = left_bound
#
# # новый вектор-градиент
#
# new_gr_f_x1_a = ';'.join(str(st) for st in gr_f_x1_a).replace('a', str(a_opt))
# new_gr_f_x1_a = [eval(line) for line in new_gr_f_x1_a.split(';')]
# print(new_gr_f_x1_a)




# # import sympy as sp
# # import numpy as np
# #
# #
# # def zoutendijk_method(f, constraints, start_point, max_iter=100):
# #     # Преобразуем входные данные в удобные формы
# #     variables = list(start_point.keys())
# #
# #     # Конвертируем начальную точку в вектор NumPy
# #     current_point = np.array([start_point[v] for v in variables])
# #
# #     # Получаем число переменных
# #     n = len(current_point)
# #
# #     # Формирование символической целевой функции и ограничений
# #     symbolic_vars = sp.symbols(variables)
# #     F = sp.sympify(f)
# #     grad_F = np.array(sp.Matrix([F.diff(var) for var in symbolic_vars]))
# #
# #     # Ограничения
# #     g_list = [sp.sympify(g) for g in constraints]
# #     grad_g = [np.array(sp.Matrix(g.diff(var) for var in symbolic_vars)) for g in g_list]
# #
# #     # Матрица Гессе целевой функции
# #     hessian_matrix = sp.hessian(F, symbolic_vars)
# #
# #     def compute_value(expr, point):
# #         return float(expr.subs({v: p for v, p in zip(symbolic_vars, point)}))
# #
# #     def gradient_at_point(grad_expr, point):
# #         return np.array([compute_value(gi, point) for gi in grad_expr])
# #
# #     iteration = 0
# #     while True and iteration < max_iter:
# #         # Текущие значения целевой функции и градиента
# #         grad_current = gradient_at_point(grad_F, current_point)
# #
# #         # Если градиент близок к нулю, останавливаемся
# #         if np.linalg.norm(grad_current) < 1e-8:
# #             break
# #
# #         # Составляем матрицу A из градиентов активных ограничений
# #         active_constraints_indices = [i for i, g in enumerate(g_list) if compute_value(g, current_point) >= 0]
# #         matrix_A = np.vstack([gradient_at_point(grad_g[i], current_point) for i in active_constraints_indices])
# #
# #         # Ищем оптимальное направление
# #         direction = None
# #         try:
# #             direction = np.linalg.solve(matrix_A.T @ matrix_A, -matrix_A.T @ grad_current)
# #         except np.linalg.LinAlgError:
# #             print("Нет подходящего направления.")
# #             break
# #
# #         # Обновление точки методом Зойтендайка
# #         step_size = 1e-3  # Можно выбрать подходящий размер шага
# #         new_point = current_point + step_size * direction
# #
# #         # Критерий остановки: изменения координат незначительны
# #         if np.linalg.norm(new_point - current_point) < 1e-6:
# #             break
# #
# #         current_point = new_point
# #         iteration += 1
# #
# #     return {var: val for var, val in zip(variables, current_point)}
# #
# #
# # # Пример использования
# # # f = "x**2 + y**2"
# # # constraints = ["x+y-1"]
# # # start_point = {"x": 0.5, "y": 0.5}
# # # result = zoutendijk_method(f, constraints, start_point)
# #
# # # Пример использования
# # f = "-6 * x1 ** 2 - x2 ** 2 + 2 * x1 * x2 + 10 * x2"
# # constraints = ["-2 * x1 - x2 - 5", "2 * x1 + x2 - 2"]
# # start_point = {"x1": 0, "x2": 4}
# # result = zoutendijk_method(f, constraints, start_point)
# #
# # print(result)

import sympy as sp
import numpy as np


# def zoutendijk_method(f, constraints, start_point, max_iter=100):
#     # Преобразуем входные данные в удобные формы
#     variables = list(start_point.keys())
#
#     # Конвертируем начальную точку в вектор NumPy
#     current_point = np.array([start_point[v] for v in variables])
#
#     # Получаем число переменных
#     n = len(current_point)
#
#     # Формирование символической целевой функции и ограничений
#     symbolic_vars = sp.symbols(variables)
#     F = sp.sympify(f)
#     grad_F = sp.Matrix([F.diff(var) for var in symbolic_vars])
#
#     # Ограничения
#     g_list = [sp.sympify(g) for g in constraints]
#     grad_g = [sp.Matrix([g.diff(var) for var in symbolic_vars]) for g in g_list]
#
#     # Матрица Гессе целевой функции
#     hessian_matrix = sp.hessian(F, symbolic_vars)
#
#     def compute_value(expr, point):
#         return float(expr.subs({v: p for v, p in zip(symbolic_vars, point)}))
#
#     def gradient_at_point(grad_expr, point):
#         return np.array([compute_value(gi, point) for gi in grad_expr])
#
#     iteration = 0
#     while True and iteration < max_iter:
#         # Текущие значения целевой функции и градиента
#         grad_current = gradient_at_point(grad_F, current_point)
#
#         # Если градиент близок к нулю, останавливаемся
#         if np.linalg.norm(grad_current) < 1e-8:
#             break
#
#         # Составляем матрицу A из градиентов активных ограничений
#         active_constraints_indices = [i for i, g in enumerate(g_list) if compute_value(g, current_point) >= 0]
#         matrix_A = np.vstack([gradient_at_point(grad_g[i], current_point) for i in active_constraints_indices])
#
#         # Ищем оптимальное направление
#         direction = None
#         try:
#             direction = np.linalg.solve(matrix_A.T @ matrix_A, -(matrix_A.T @ grad_current[:, None]))
#         except np.linalg.LinAlgError:
#             print("Нет подходящего направления.")
#             break
#
#         # Обновление точки методом Зойтендайка
#         step_size = 1e-3  # Можно выбрать подходящий размер шага
#         new_point = current_point + step_size * direction
#
#         # Критерий остановки: изменения координат незначительны
#         if np.linalg.norm(new_point - current_point) < 1e-6:
#             break
#
#         current_point = new_point
#         iteration += 1
#
#     return {var: val for var, val in zip(variables, current_point)}
#
#
# # Пример использования
# f = "x**2 + y**2"
# constraints = ["x+y-1"]
# start_point = {"x": 0.5, "y": 0.5}
# result = zoutendijk_method(f, constraints, start_point)
#
# print(result)


# def zoutendijk_method(f, constraints, start_point, max_iter=100):
#     # Преобразуем входные данные в удобные формы
#     variables = list(start_point.keys())
#
#     # Конвертируем начальную точку в вектор NumPy
#     current_point = np.array([start_point[v] for v in variables])
#
#     # Получаем число переменных
#     n = len(current_point)
#
#     # Формирование символической целевой функции и ограничений
#     symbolic_vars = sp.symbols(variables)
#     F = sp.sympify(f)
#     grad_F = sp.Matrix([F.diff(var) for var in symbolic_vars])
#
#     # Ограничения
#     g_list = [sp.sympify(g) for g in constraints]
#     grad_g = [sp.Matrix([g.diff(var) for var in symbolic_vars]) for g in g_list]
#
#     # Матрица Гессе целевой функции
#     hessian_matrix = sp.hessian(F, symbolic_vars)
#
#     def compute_value(expr, point):
#         return float(expr.subs({v: p for v, p in zip(symbolic_vars, point)}))
#
#     def gradient_at_point(grad_expr, point):
#         return np.array([compute_value(gi, point) for gi in grad_expr]).squeeze()
#
#     iteration = 0
#     while iteration < max_iter:
#         # Текущие значения целевой функции и градиента
#         grad_current = gradient_at_point(grad_F, current_point)
#
#         # Если градиент близок к нулю, останавливаемся
#         if np.linalg.norm(grad_current) < 1e-8:
#             break
#
#         # Составляем матрицу A из градиентов активных ограничений
#         active_constraints_indices = [i for i, g in enumerate(g_list) if compute_value(g, current_point) >= 0]
#         if len(active_constraints_indices) == 0:
#             print("Нет активных ограничений.")
#             break
#
#         matrix_A = np.vstack([gradient_at_point(grad_g[i], current_point) for i in active_constraints_indices])
#
#         # Ищем оптимальное направление
#         direction = None
#         try:
#             # direction = np.linalg.solve(matrix_A.T @ matrix_A, -(matrix_A.T @ grad_current))
#             direction = np.linalg.solve(matrix_A.T @ matrix_A, -(matrix_A.T @ grad_current[:, None]))
#         except np.linalg.LinAlgError:
#             print("Нет подходящего направления.")
#             break
#
#         # Обновление точки методом Зойтендайка
#         step_size = 1e-3  # Можно выбрать подходящий размер шага
#         new_point = current_point + step_size * direction.flatten()
#
#         # Критерий остановки: изменения координат незначительны
#         if np.linalg.norm(new_point - current_point) < 1e-6:
#             break
#
#         current_point = new_point
#         iteration += 1
#
#     return {var: val for var, val in zip(variables, current_point)}
#
#
# f = "x**2 + y**2"
# constraints = ["x+y-1"]
# start_point = {"x": 0.5, "y": 0.5}
# result = zoutendijk_method(f, constraints, start_point)
#
# print(result)

import sympy as sp
import numpy as np

# def zoutendijk_method(f, constraints, start_point, max_iter=100):
#     # Преобразуем входные данные в удобные формы
#     variables = list(start_point.keys())
#
#     # Конвертируем начальную точку в вектор NumPy
#     current_point = np.array([start_point[v] for v in variables])
#
#     # Получаем число переменных
#     n = len(current_point)
#
#     # Формирование символической целевой функции и ограничений
#     symbolic_vars = sp.symbols(variables)
#     F = sp.sympify(f)
#     grad_F = sp.Matrix([F.diff(var) for var in symbolic_vars])
#
#     # Ограничения
#     g_list = [sp.sympify(g) for g in constraints]
#     grad_g = [sp.Matrix([g.diff(var) for var in symbolic_vars]) for g in g_list]
#
#     # Матрица Гессе целевой функции
#     hessian_matrix = sp.hessian(F, symbolic_vars)
#
#     def compute_value(expr, point):
#         return float(expr.subs({v: p for v, p in zip(symbolic_vars, point)}))
#
#     def gradient_at_point(grad_expr, point):
#         return np.array([compute_value(gi, point) for gi in grad_expr]).squeeze()
#
#     iteration = 0
#     while iteration < max_iter:
#         # Текущие значения целевой функции и градиента
#         grad_current = gradient_at_point(grad_F, current_point)
#
#         # Если градиент близок к нулю, останавливаемся
#         if np.linalg.norm(grad_current) < 1e-8:
#             break
#
#         # Составляем матрицу A из градиентов активных ограничений
#         active_constraints_indices = [i for i, g in enumerate(g_list) if compute_value(g, current_point) >= 0]
#         if len(active_constraints_indices) == 0:
#             print("Нет активных ограничений.")
#             break
#
#         matrix_A = np.vstack([gradient_at_point(grad_g[i], current_point) for i in active_constraints_indices])
#
#         # Проверка матрицы на пустоту и корректировку формы
#         if matrix_A.size == 0:
#             print("Матрица ограничений пуста!")
#             break
#         elif matrix_A.ndim == 1:
#             matrix_A = matrix_A[:, None]
#
#         # Ищем оптимальное направление
#         direction = None
#         try:
#             # Преобразуем grad_current в вектор-столбец
#             direction = np.linalg.solve(matrix_A.T @ matrix_A, -(matrix_A.T @ grad_current[:, None])).flatten()
#         except np.linalg.LinAlgError:
#             print("Нет подходящего направления.")
#             break
#
#         # Обновление точки методом Зойтендайка
#         step_size = 1e-3  # Можно выбрать подходящий размер шага
#         new_point = current_point + step_size * direction
#
#         # Критерий остановки: изменения координат незначительны
#         if np.linalg.norm(new_point - current_point) < 1e-6:
#             break
#
#         current_point = new_point
#         iteration += 1
#
#     return {var: val for var, val in zip(variables, current_point)}
#
# # Пример использования
# f = "x**2 + y**2"
# constraints = ["x+y-1", '-x-y-2']
# start_point = {"x": 0.5, "y": 0.5}
# result = zoutendijk_method(f, constraints, start_point)
#
# print(result)

import sympy as sp
import numpy as np


def zoutendijk_method(f, constraints, x0, max_iter=1000):
    vars = list(x0.keys())

    cur_point = np.array([x0[v] for v in vars])

    sym_vars = sp.symbols(vars)
    F = sp.sympify(f)
    grad_f = sp.Matrix([F.diff(var) for var in sym_vars])

    g_list = [sp.sympify(g) for g in constraints]
    grad_g = [sp.Matrix([g.diff(var) for var in sym_vars]) for g in g_list]

    # Функция, подставляющая значения в выражение
    def compute_value(expr, point):
        return float(expr.subs({v: p for v, p in zip(sym_vars, point)}))

    def gradient_in_point(grad_expr, point):
        return np.array([compute_value(gi, point) for gi in grad_expr]).squeeze()

    iteration = 0
    while iteration < max_iter:
        # Текущие значения целевой функции и градиента
        cur_grad = gradient_in_point(grad_f, cur_point)

        # Условие остановки
        if np.linalg.norm(cur_grad) < 1e-8:
            break

        # Составляем матрицу A из градиентов активных ограничений
        active_constraints_indices = [i for i, g in enumerate(g_list) if compute_value(g, cur_point) >= 0] # проверка "активности ограничения"
        if len(active_constraints_indices) == 0:
            print("Нет активных ограничений.")
            break

        # вертикальная сборка векторов-столбцов активных ограничений
        matrix_A = np.vstack([gradient_in_point(grad_g[i], cur_point) for i in active_constraints_indices])

        # Пустая ли матрица?
        if matrix_A.size == 0:
            print("Матрица ограничений пуста!")
            break
        # Добиваем матрицу значениями при необходимости
        elif matrix_A.ndim == 1:
            matrix_A = matrix_A[:, None]

        # Ищем оптимальное направление
        direction = None
        try:
            # matrix_A.T @ cur_grad[:, None] - вектор-вклад активных ограничений в определение направления
            # matrix_A.T @ matrix_A - матрица нормальных уравнений (учет ограничений)
            direction = np.linalg.solve(matrix_A.T @ matrix_A, -(matrix_A.T @ cur_grad[:, None])).flatten()
        except np.linalg.LinAlgError:
            print("Нет подходящего направления.")
            break

        # Обновление точки
        step = 1e-3
        new_point = cur_point + step * direction

        # Достигли нужного эпсилона
        if np.linalg.norm(new_point - cur_point) < 1e-6:
            break

        cur_point = new_point
        iteration += 1

    return {var: val for var, val in zip(vars, cur_point)}


f = "x**2 + y**2"
constraints = ["x+y-1", "-2*x-2*y+1", 'x-2*y+10']
x0 = {"x": 0, "y": 1}
result = zoutendijk_method(f, constraints, x0)

print(f'f* = {result['x'] ** 2 + result['y'] ** 2}')
print(f'В точке ({result['x']}; {result['y']})')