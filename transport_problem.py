def get_task(filename):
    with open(filename, 'rt', encoding='utf-8') as f:
        data = f.read().splitlines()
        print(data)
        rows, cols = [int(x) for x in data[0].split('x')]
        transport_table = [[int(x) for x in line.split()]
                           for line in data[1:rows + 1]]
        reserves = [int(x) for x in data[-2].split()]
        needs = [int(x) for x in data[-1].split()]
        return rows, cols, transport_table, reserves, needs


# Матрица цен (стоимости)
price_matrix = []
n, m, list_ai, list_bj = 0, 0, [], []
answer = input('Ручной ввод (h) / Ввод из файла (f): ')
if answer == 'h':
    n = int(input("Строки: "))
    m = int(input("Столбцы: "))
    for i in range(n):
        row = []
        for j in range(m):
            price = input(f"Цена {i+1} {j+1}: ")
            row.append(price)
        price_matrix.append(row)
    print(price_matrix)

    # Запасы и потребности
    list_ai = []
    for i in range(n):
        ai = int(input(f"Запасы продукта {i+1}: "))
        list_ai.append(ai)

    list_bj = []
    for j in range(m):
        bj = int(input(f"Потреба продукта {j+1}: "))
        list_bj.append(bj)

else:
    n, m, price_matrix, list_ai, list_bj = get_task('data/task_cl_wk.txt')

#Приведение к закрытому типу
if sum(list_ai) > sum(list_bj):
    b_new = sum(list_ai) - sum(list_bj)
    list_bj.append(b_new)
    m += 1
    for i in range(n):
        price_matrix[i].append(0)

elif sum(list_ai) < sum(list_bj):
    a_new = sum(list_bj) - sum(list_ai)
    list_ai.append(a_new)
    n += 1
    zeros = []
    for i in range(m):
        zeros.append(0)
    price_matrix.append(zeros)
print(price_matrix)
bas_var = n + m - 1

# Транспортная таблица (пустая)
trans_table = []
for i in range(n):
    row = []
    for j in range(m):
        row.append("-")
    trans_table.append(row)

# Метод СЗ угла - можно просто через цикл, он будет всегда оказываться слева сверху по порядку обхода
tmp_list_ai = list_ai[:]
tmp_list_bj = list_bj[:]
for i in range(n):
    for j in range(m):
        if trans_table[i][j] == "-":
            ai = tmp_list_ai[i]
            bj = tmp_list_bj[j]
            bas = min(ai, bj)
            trans_table[i][j] = bas
            tmp_list_ai[i] = ai - bas
            tmp_list_bj[j] = bj - bas
            if ai == bas:
                for k in range(m):
                    if trans_table[i][k] == "-":
                        trans_table[i][k] = "x"
            elif bj == bas:
                for k in range(n):
                    if trans_table[k][j] == "-":
                        trans_table[k][j] = "x"

print(trans_table)

# Поиск индексов элементов в порядке возрастания их стоимости
# all_prices = [price_matrix[i][j] for i in range(n) for j in range(m)]
# all_prices.sort()
# pos = 0
# indexes = []
# stop = False
# while not stop:
#     for i in range(n):
#         cur_elem = all_prices[pos]
#         if cur_elem in price_matrix[i]:
#             indexes.append((i, price_matrix[i].index(cur_elem)))
#             pos += 1
#             if pos == m * n:
#                 stop = True
#                 break

# # Метод минимальной стоимости
# tmp_list_ai = list_ai[:]
# tmp_list_bj = list_bj[:]
# for index in indexes:
#     i, j = index
#     if trans_table[i][j] == "-":
#         ai = tmp_list_ai[i]
#         bj = tmp_list_bj[j]
#         bas = min(ai, bj)
#         trans_table[i][j] = bas
#         tmp_list_ai[i] = ai - bas
#         tmp_list_bj[j] = bj - bas
#         if ai == bas:
#             for k in range(m):
#                 if trans_table[i][k] == "-":
#                     trans_table[i][k] = "x"
#         elif bj == bas:
#             for k in range(n):
#                 if trans_table[k][j] == "-":
#                     trans_table[k][j] = "x"
#
# print(trans_table)

while True:
    # Метод потенциалов (создание ui и vj)
    list_ui = []
    list_vj = []
    list_ui.append(0)
    for i in range(n - 1):
        list_ui.append("-")
    for j in range(m):
        list_vj.append("-")

    while True:
        for i in range(n):
            if list_ui[i] != "-":
                for j in range(m):
                    if trans_table[i][j] != "x":
                        list_vj[j] = int(price_matrix[i][j]) - list_ui[i]
        for j in range(m):
            if list_vj[j] != "-":
                for i in range(n):
                    if trans_table[i][j] != "x":
                        list_ui[i] = int(price_matrix[i][j]) - list_vj[j]
        if "-" not in list_ui and "-" not in list_vj:
            break
    print(list_ui)
    print(list_vj)

    # Список нехороших клеток
    bad_cells = []

    # "Ведущий" элемент, с которого начнется цикл пересчета; delta price - разница псевдостоимости и стоимости, если она > 0, т. е. псевдо ст. > ст., то условие не выполнено и нужно пересчитывать
    lead_deltaprice = 0
    lead_row = 0
    lead_col = 0
    for i in range(n):
        for j in range(m):
            if str(trans_table[i][j]) in '-x':
                ps_price = list_ui[i] + list_vj[j]
                if ps_price > price_matrix[i][j]:
                    bad_cells.append((i, j))

    if len(bad_cells) == 0:
        print("Оптимальненько")
        # Поиск L*
        l_optimal = 0
        for i in range(n):
            for j in range(m):
                if str(trans_table[i][j]).isdigit():
                    l_optimal += trans_table[i][j] * price_matrix[i][j]
        print(f'L* = {l_optimal}')
        # Оптимальное значение найдено => завершение работы алгоритма
        break
    else:
        # Цикл пересчета
        # Поиск клеток, по которым пройдет цикл
        print('Зашли в цикл пересчета')
        cycle = []
        stop = False
        for bc in bad_cells:
            if stop:
                break
            bad_cell = bc
            for vert in range(n):
                if cycle:
                    stop = True
                    break
                if str(trans_table[vert][bad_cell[1]]).isdigit():
                    for gor in range(m):
                        if str(trans_table[bad_cell[0]][gor]).isdigit():
                            if (vert, gor) != bad_cell:
                                if str(trans_table[vert][gor]).isdigit():
                                    cycle = [bad_cell, (vert, bad_cell[1]), (vert, gor), (bad_cell[0], gor)]
                                    break

        d = min(trans_table[cycle[1][0]][cycle[1][1]], trans_table[cycle[3][0]][cycle[3][1]])
        # Пересчет иксов
        count = 0
        for i, j in cycle:
            if count % 2 == 0:
                if not str(trans_table[i][j]).isdigit():
                    trans_table[i][j] = d
                else:
                    trans_table[i][j] += d
            else:
                if trans_table[i][j] == d:
                    trans_table[i][j] = 'x'
                else:
                    trans_table[i][j] -= d
            count += 1

        print(trans_table)
