########################Task_2_2###############################
# results for m_1 and m_2 are equal
###############################################################

import numpy as np
import matplotlib.pyplot as plt

def create_matrix(n):
    arr = []
    for i in range(n):
        arr.append(n*[0])

    for i in range(n):
        arr[i][i] = -2
    for i in range(n - 1):
        arr[i][i + 1] = 1
    for i in range(1, n):
        arr[i][i - 1] = 1

    arr = np.array(arr)
    return arr

def create_inverse_matirix(n):
    return np.linalg.inv(create_matrix(n))

def m_1(arr):
    norma_str = max(list([np.sum(abs(arr), axis=0)][0]))
    print(list([np.sum(abs(arr), axis=0)]))
    return norma_str

def m_2(arr):
    norma_str = max([np.sum(abs(arr), axis=1)][0])
    return norma_str

def m_3(arr):
    norm = max(abs(np.linalg.eig(np.dot(arr, np.transpose(arr)))[0]))**(0.5)
    return norm

n = int(input("n = "))
x = np.arange(1, n + 1)

nu_m_1 = np.empty(n)
nu_m_2 = np.empty(n)
nu_m_3 = np.empty(n)

for i in x:
    matrix = create_matrix(i)
    inverse_matrix = create_inverse_matirix(i)
    nu_m_1[i - 1] = m_1(matrix) * m_1(inverse_matrix)
    nu_m_2[i - 1] = m_2(matrix) * m_2(inverse_matrix)
    nu_m_3[i - 1] = m_3(matrix) * m_3(inverse_matrix)

plt.plot(x, nu_m_1, color='red', marker='.')
plt.plot(x, nu_m_2, color='green', marker='.')
plt.plot(x, nu_m_3, color='blue', marker='.')
plt.grid
plt.xlabel('n')
plt.ylabel('Число обусловленности')

plt.show()
