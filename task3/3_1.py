import numpy as np
import matplotlib.pyplot as plt

def decLU(A):
    size = A.shape[0]
    U = np.zeros((size, size))
    L = np.zeros((size, size))

    for i in np.arange(0, size, 1):
        for j in np.arange(0, size, 1):
            if i == 0:
                U[i][j] = A[i][j]
                L[j][i] = A[j][i] / U[0][0]
            else:
                S = 0.0
                for k in np.arange(0, i - 1, 1):
                    S += L[i][k] * U[k][i]

                U[i][i] = A[i][i] - S
                S = 0.0

                for k in np.arange(0, i, 1):
                    S += L[i][k] * U[k][j]

                U[i][j] = A[i][j] - S
                S = 0.0

                for k in np.arange(0, i, 1):
                    S += L[j][k] * U[k][i]

                L[j][i] = (A[j][i] - S) / U[i][i]

    return U, L

def Solve(A, f):
    size = A.shape[0]

    y = np.zeros(size)
    x = np.zeros(size)
    U, L = decLU(A)

    for k in np.arange(0, size, 1):
        y[k] = f[k] - np.dot(L[k][0:k], y[0:k])

    for k in np.arange(size - 1, -1, -1):
        x[k] = (y[k] - np.dot(U[k][k+1:size], x[k+1:size])) / U[k][k]

    return x

def create_matrix(n):
    arr = []
    for i in range(n - 1):
        arr.append(n * [0])
    
    arr.append(n * [1])

    for i in range(n - 1):
        for j in range(n):
            if i < j:
                arr[i][j] = -1
            elif i > j:
                arr[i][j] = 0
            else:
                arr[i][j] = 1

    arr = np.array(arr)
    return arr

n = int(input("n = "))

A = create_matrix(n) 
f = np.array(n * [1])

res = Solve(A, f)

x = np.arange(3, n + 1)
cond_nums = [0] * (n - 2)

for i in range(3, n + 1):
    cond_nums[i - 3] = np.linalg.cond(create_matrix(i))

plt.plot(x, cond_nums, color='red', marker='.')
plt.grid()
plt.title(r'Зависимость числа обусловленности от размерности n')
plt.xlabel('n')
plt.ylabel('Число обусловленности')
plt.show()
