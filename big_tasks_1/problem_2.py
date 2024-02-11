import numpy as np
import matplotlib.pyplot as plt
import problem_1


def create_matrix(n: int) -> np.array:
    A = np.zeros((n, n))
    for i in range (n):
        for j in range(n):
            if i == j:
                A[i, j] =  2.0 + (np.float64(i) / np.float64(n)) ** 2.0
            elif j == i - 1 or j == i + 1:
                A[i, j] =  -1.0
            elif (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                A[i, j] = -1.0
            else:
                A[i, j] = 0.0

    return A

def create_f(n: int) -> np.array:
    f = np.zeros(n)
    for i in range(n):
        f[i] = (1.0 + n**2.0 * np.sin(np.pi / n) ** 2.0) * np.sin( (2.0 * np.pi * i) / np.float64(n))

    return f 

def gershgorin_circles(a: np.array):
    for i, row in enumerate(a):
        a_ii = a[i, i]
        r_i = np.abs(row).sum() - a_ii
        print(f"D({a_ii:5.3f}; {r_i:5.3f})")

def LUP_decomposition(A):
    n = A.shape[0]
    P = np.eye(n)
    L = np.zeros((n, n))
    U = np.copy(A)

    for j in range(n):
        max_row = np.argmax(np.abs(U[j:, j])) + j

        if j != max_row:
            P[[j, max_row]] = P[[max_row, j]]
            L[[j, max_row], :j] = L[[max_row, j], :j]
            U[[j, max_row], j:] = U[[max_row, j], j:]

        L[j, j] = 1.0
        for i in range(j + 1, n):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]

    return P, L, U

def LUP_solve(A: np.array, f: np.array):
    P, L, U = LUP_decomposition(A)

    z = np.matmul(P, f)
    x = problem_1.LU_solve(L, U, z)

    return x

def get_eigvals_krylov(A):
    size = A.shape[0]
    y_matrix = np.zeros((size, size))
    y = np.ones(n)

    for i in range(size - 1, -1, -1):
        y_matrix[:, i] = y
        y = np.matmul(A, y)

    ps = LUP_solve(y_matrix, y)

    coeffs = np.concatenate([np.ones(1), -1.0 * ps])

    return np.array(sorted(np.roots(coeffs)))

def solve(A, f, tau, eps=1e-6, iter_limit = 10000):
    size = A.shape[0]
    tau_f = f * tau
    cur = np.zeros(size)
    b = np.eye(size) - tau * A
    err_list = []

    for i in range(iter_limit):
        cur = np.matmul(b, cur) + tau_f
        err = f - np.matmul(A, cur)
        norm_err = problem_1.m_3(err)
        err_list.append(norm_err)

        if norm_err < eps:
            break

    solution = cur 
    return solution, err_list

def output_solve(A, f, tau, eps=1e-6):
    solution, errors = solve(A, f, tau=tau, eps=eps)
    exact_sol = np.linalg.solve(A, f)
    delta = solution - exact_sol
    print("---------------------------------------------------")
    print("solution")
    print(f"tau = {tau}; res = {solution}")
    print(f"|numpy_sol - sol|_3 = {problem_1.m_3(delta)}")
    print("---------------------------------------------------")

    num_errors = np.arange(len(errors))
    line, = plt.plot(num_errors, errors, marker='.')
    line.set_label(f"tau = {tau}")

if __name__ == '__main__':
    n = 6
    A = create_matrix(n)
    f = create_f(n)

    print("---------------------------------------------------")
    print("eigenvalue estimation using Gershgorin circles")
    gershgorin_circles(A)
    print("---------------------------------------------------")

    krylov_eigvals = get_eigvals_krylov(A)
    krylov_eigvals.sort()
    numpy_eigvals = np.linalg.eigvals(A)
    numpy_eigvals.sort()

    print("---------------------------------------------------")
    print("eigenvals (krylov vs numpy)")
    for i in range (len(krylov_eigvals)):
        l_k = krylov_eigvals[i]
        l_n = numpy_eigvals[i]
        print(f"lambda_{i}: {l_k:10.6f} vs {l_n:10.6f}; delta = {abs(l_n - l_k)}")
    print("---------------------------------------------------")

    tau_opt_krylov = 2.0 / (krylov_eigvals.min() + krylov_eigvals.max())
    tau_opt_numpy = 2.0 / (numpy_eigvals.min() + numpy_eigvals.max())

    print(f"tau_opt_numpy  = {tau_opt_numpy}")
    print(f"tau_opt_krylov = {tau_opt_krylov}")

    output_solve(A, f, tau_opt_numpy)
    output_solve(A, f, tau_opt_krylov)
    output_solve(A, f, tau=0.4)
    output_solve(A, f, tau=0.5)

    plt.yscale("log")
    plt.ylabel('mismatch rate')
    plt.xlabel('number of iterations')
    plt.grid()
    plt.legend()
    plt.show()
