import numpy as np
import math
import matplotlib.pyplot as plt
import scipy

import problem_3

def getLejanInit(n: int) -> np.array:
    theta_n_k = lambda k: np.pi * (n - k + 0.75) / (n + 0.5)
    get_xk = lambda k: (1.0 - 1 / (8.0 * n ** 2.0) + 1 / (8.0 * n ** 3.0) - 1 / (384.0 * n ** 4.0) * (39.0 - 28.0 / (np.sin(theta_n_k(k)) ** 2.0))) * np.cos(theta_n_k(k))
    return np.sort(np.array([get_xk(k) for k in range(1, n + 1)]))

def function_2(X : float) -> float:
    return math.log(100.0 - X) / (10.0 - math.sqrt(X))

def calculateSimpson(Functions : list, a : float, b : float) -> float:
    Int = 0
    N = len(Functions)
    k = int(N / 2)
    h = (b - a) / N
    for i in range(1, k):
        F1 = Functions[2 * i]
        F2 = Functions[2 * i - 1]
        F3 = Functions[2 * i - 2]
        Int += h / 3.0 * (F1 + 4 * F2 + F3)
    return Int

def getLejanPol_v2(n: int) -> np.array:
    poly_x_sq_minus_one = np.array([1.0, 0, -1.0])
    poly_to_power = np.polynomial.polynomial.polypow(poly_x_sq_minus_one, n)
    poly_der = np.polyder(poly_to_power, n)

    return poly_der / (math.gamma(n + 1.0) * 2.0 ** n)

def getLejanZeros_v2(n: int) -> list:
    poly = getLejanPol_v2(n)
    polyder = np.polyder(poly)
    xs = getLejanInit(n)
    
    def f(x):
        return np.polyval(poly, x)
    def f_der(x):
        return np.polyval(polyder, x)
    
    res = []
    for x in xs:
        eps = 1e-3
        delta = 1
        while (abs(delta) > eps):
            delta = f(x) / f_der(x)
            x -= delta

            res.append(x)

    return res

def getLejanZeros(N : int) -> list:
    def getLejanPol(X : float, N : int) -> float:
        if (N == 0):
            return 1
        if (N == 1):
            return X
        
        return (2.0 * N + 1) * X * getLejanPol(X, N - 1) / (N + 1) - N * getLejanPol(X, N - 2) / (N + 1)

    def getLejanDerr(X : float, N : int) -> float:
        return N * (getLejanPol(X, N - 1) - X * getLejanPol(X, N)) / (1 - X * X)

    def getNextIter(Xk : float, P : float, P1 : float) -> float:
        return Xk - P / P1

    res = []
    eps = 1e-3
    for i in range(1, N + 1):
        Xk = math.cos(math.pi * (4 * i - 1) / (4 * N + 2))
        Xk1 = getNextIter(Xk, getLejanPol(Xk, N), getLejanDerr(Xk, N))
        while (abs(Xk - Xk1) > eps):
            Xk = Xk1
            Xk1 = getNextIter(Xk, getLejanPol(Xk, N), getLejanDerr(Xk, N))
        res.append(Xk1)

    return res 

def changeVars(Start : float, Stop : float, Vars : list) -> list:
    HalfSum = (Start + Stop) / 2.0
    HalfDiff = (Stop - Start) / 2.0
    NewVars = [HalfSum + HalfDiff * T for T in Vars]
    return NewVars

def calculateWithGaussQuadrature(F, a : float, b : float, N : int) -> float:
    Int = 0
    Int_2 = 0
    NodesT = getLejanZeros(N)
    NodesT_2 = getLejanZeros_v2(N)
    # print("getLejan v2: ", getLejanZeros_v2(N))
    # print("getLejan v1: ", getLejanZeros(N))
    NodesX = changeVars(a , b, NodesT)
    NodesX_2 = changeVars(a , b, NodesT_2)
    for k in range(1, N + 1):
        Args = np.linspace(a, b, 10000)
        BaseLagranValues = [problem_3.get_l(i, k - 1, NodesX) for i in Args]
        Ck = calculateSimpson(BaseLagranValues, a, b)
        Fk = F(NodesX[k - 1])
        Int += Ck * Fk

        BaseLagranValues_2 = [problem_3.get_l(i, k - 1, NodesX_2) for i in Args]
        Ck_2 = calculateSimpson(BaseLagranValues_2, a, b)
        Fk_2 = F(NodesX_2[k - 1])
        Int_2 += Ck_2 * Fk_2
    return Int, Int_2
    

if __name__ == '__main__':
    a = 0
    b = 10

    ArrayN = np.arange(3, 10)

    Exact = scipy.integrate.quad(function_2, a, b)
    print("exact I =", Exact[0])

    Errors =[]
    Errors_2 =[]
    for N in ArrayN:
        Gauss, Gauss_v2 = calculateWithGaussQuadrature(function_2, a, b, N)
        print("N =", N, ", I =", format(Gauss, '.10f'))
        Errors.append(abs(Gauss - Exact[0]) * 100 / Exact[0])
        Errors_2.append(abs(Gauss_v2 - Exact[0]) * 100 / Exact[0])

    plt.figure(figsize = (10, 10))
    plt.title("зависимость относительной ошибки интегрирования от количества узлов")
    plt.scatter(ArrayN, Errors, label = "v1")
    plt.scatter(ArrayN, Errors_2, label = "v2")
    plt.xlabel('n')
    plt.ylabel('error, %')
    plt.grid()
    plt.legend()
    plt.show()
