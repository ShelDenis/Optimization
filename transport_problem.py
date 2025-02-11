import random
# Матрица цен (стоимости)
price_matrix = []
n = int(input("Строки: "))
m = int(input("Столбцы: "))
for i in range(n):
    row = []
    for j in range(m):
        price = input(f"Цена {i+1} {j+1}: ")
        row.append(price)
        # row.append(random.randint(0, 10))
    price_matrix.append(row)
print(price_matrix)
# Готовая матрица
# price_matrix = [['40', '36', '9', '20', '0'], ['26', '11', '22', '26', '0'], ['6', '3', '12', '3', '0'], ['5', '37', '33', '26', '0']]
# Ответ: [[24, 'x', 'x', 'x', 'x'], [11, 29, 2, 'x', 'x'], ['x', 'x', 19, 4, 'x'], ['x', 'x', 'x', 31, 5]]

# Запасы и потребности
list_ai = []
for i in range(n):
    ai = int(input(f"Запасы продукта {i+1}: "))
    list_ai.append(ai)
# list_ai = [24, 42, 23, 36]

list_bj = []
for j in range(m):
    bj = int(input(f"Потреба продукта {j+1}: "))
    list_bj.append(bj)
# list_bj = [35, 29, 21, 35, 5]

#Делаем открытый тип (задание называется Открытый тип)
# if sum(list_ai)==sum(list_bj):
#     el = list_bj.pop()
#     list_bj.append(el+1)
# print(list_ai)
# print(list_bj)

#Приведение к закрытому типу
if sum(list_ai)>sum(list_bj):
    b_new = sum(list_ai)-sum(list_bj)
    list_bj.append(b_new)
    m+=1
    for i in range(n):
        price_matrix[i].append(0)
elif sum(list_ai)<sum(list_bj):
    a_new = sum(list_bj)-sum(list_ai)
    list_ai.append(a_new)
    n+=1
    zeros = []
    for i in range(m):
        zeros.append(0)
    price_matrix.append(zeros)
print(price_matrix)
bas_var = n+m-1
# Транспортная таблица (пустая)
trans_table = []
for i in range(n):
    row = []
    for j in range(m):
        row.append("-")
    trans_table.append(row)
print(trans_table)
#Метод СЗ угла - можно просто через цикл, он будет всегда оказываться слева сверху по порядку обхода
for i in range(n):
    for j in range(m):
        if trans_table[i][j] == "-":
            ai = list_ai[i]
            bj = list_bj[j]
            bas = min(ai, bj)
            trans_table[i][j] = bas
            list_ai[i] = ai - bas
            list_bj[j] = bj - bas
            if ai == bas:
                for k in range(m):
                    if trans_table[i][k] == "-":
                        trans_table[i][k] = "x"
            elif bj == bas:
                for k in range(n):
                    if trans_table[k][j] == "-":
                        trans_table[k][j] = "x"
                
print(trans_table)

# Метод потенциалов (создание ui и vj)
list_ui = []
list_vj = []
list_ui.append(0)
for i in range(n-1):
    list_ui.append("-")
for j in range(m):
    list_vj.append("-")

while(True):
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
# "Ведущий" элемент, с которого начнется цикл пересчета; delta price - разница псевдостоимости и стоимости, если она > 0, т. е. псевдо ст. > ст., то условие не выполнено и нужно пересчитывать
lead_deltaprice = 0
lead_row = 0
lead_col = 0
for i in range(n):
    for j in range(m):
        ps_price = list_ui[i] + list_bj[j]
        if ps_price - int(price_matrix[i][j]) > lead_deltaprice:
            lead_deltaprice = ps_price - int(price_matrix[i][j])
            lead_row = i
            lead_col = j

if lead_deltaprice == 0:
    print("Оптимальненько")
# else:

# Далее каким-то образом идет по базисным, поворачивая на 90 градусов и образуя цикл (не ясно, может ли цикл быть не квадратом, а больше)
######
#    #
#    #
#    ######
#         #
###########
# типо такого

# Когда сделаем цикл пересчета, нужно будет всё, начиная с метода СЗ, завернуть в цикл, пока не выполнится условие (псевдо ст. <= ст.)