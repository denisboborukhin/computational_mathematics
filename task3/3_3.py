import numpy as np
import matplotlib.pyplot as plt

def create_system(alpha, n):
    A = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 2
            if j - i == 1:
                A[i, j] = - 1 - alpha
            if j - i == -1:
                A[i, j] = - 1 + alpha

    f = np.zeros(shape=(n, 1))
    f[0] = 1 - alpha
    f[n - 1] = 1 + alpha

    return (A, f)

def decomposing(A):
    n = A.shape[0]
    L = np.zeros(shape=(n, n))
    D = np.zeros(shape=(n, n))
    U = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = A[i, j]
            if i < j:
                U[i, j] = A[i, j]
            if i == j:
                D[i, j] = A[i, j]

    return (L, D, U)

def Euclidian_norm(A):
    return np.sqrt(np.sum(np.multiply(A, A), axis=(0)))

def Seidel_iterations(alpha, n):
    u_old = np.zeros(shape=(n, 1))
    u_old[:] = 0 

    A, f = create_system(alpha, n)
    L, D, U = decomposing(A)
    u_new = -np.matmul(np.linalg.inv(L + D), U) @ u_old + np.matmul(np.linalg.inv(L + D), f)

    eps = 0.01
    iters = 0
    while np.linalg.norm(u_new - u_old) > eps and iters < 10000:
        u_old = u_new
        u_new = -np.matmul(np.linalg.inv(L + D), U) @ u_old + np.matmul(np.linalg.inv(L + D), f)
        iters += 1

    return iters - 1

num_alphs = 50
alphs = np.linspace(0, 1, num_alphs)
iters = []

for alph in alphs:
    iters.append(Seidel_iterations(alph, 15))

plt.plot(alphs, iters, color='red', marker='.')
plt.grid()
plt.title(r'Зависимость числа итераций от $\alpha$')
plt.xlabel(r'$\alpha$')
plt.ylabel('число итераций')
plt.show()
