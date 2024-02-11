import numpy as np

def create_matrix(n: int) -> np.array:
    A = np.zeros((n, n))
    for i in range (n):
        for j in range(n):
            A[i, j] = 10.0 / (i + j + 1)
    return A

def create_f(n: int) -> np.array:
    return np.sum(create_matrix(n), axis=0)

def cholesky_decomposition(matrix):
    lower = np.zeros_like(matrix)
    size = matrix.shape[0]

    for i in range(size):
        i_row = lower[i, :]

        for j in range(i + 1):
            j_row = lower[j, :]
            dot_product = 0
            for k in range(size):
                dot_product += i_row[k] * j_row[k]

            lower[i, j] = np.sqrt(matrix[j, j] - dot_product) if i == j else (1 / lower[j, j]) * (matrix[i, j] - dot_product)

    return lower

def LU_solve(L, U, f):
    size = L.shape[0]

    y = np.zeros(size)
    for i in range(size):
        tmp = f[i]
        for j in range(i):
            tmp -= L[i, j] * y[j]
        y[i] = tmp / L[i, i]

    x = np.zeros(size)
    for i in range(size - 1, -1, -1):
        tmp = y[i]
        for j in range(i + 1, size):
            tmp -= U[i, j] * x[j]
        x[i] = tmp / U[i, i]

    return x

def m_1(arr):
    norma = max(abs(arr))
    return norma

def m_2(arr):
    norma = sum(abs(arr))
    return norma

def m_3(arr):
    norma = sum(abs(arr) ** 2) ** 0.5
    return norma

if __name__ == '__main__':
    n = 6
    A = create_matrix(n)
    f = create_f(n)
    L = cholesky_decomposition(A)

    U = np.transpose(L)
    x = LU_solve(L, U, f)

    print(f"L = \n{L}")
    print(f"x = \n{x}")

    delta = np.linalg.solve(A, f) - x

    print(f"delta_1 = {m_1(delta)}")
    print(f"delta_2 = {m_2(delta)}")
    print(f"delta_3 = {m_3(delta)}")
