import numpy as np
import math
import matplotlib.pyplot as plt

import problem_3
import problem_4

def g(X : float) -> float:
    return 1

def f(X : float) -> float:
    return math.cos(X * math.pi)

def K(X : float, S : float) -> float:
    return 0.2 / (0.04 + (X - S)**2)

def calculateTrapezoidByPoints(Functions : list, a : float, b : float) -> float:
    # Считает интеграл от поточечно заданной функции со значениями Functions[i] на отрезке [a, b] методом трапеций
    Int = 0
    N = len(Functions)
    h = (b - a) / N
    for i in range(1, N):
        F1 = Functions[i]
        F2 = Functions[i - 1]
        Int += h / 2.0 * (F1 + F2)
    return Int

def LUdecomposition(A : list) -> (list, list):
    # Разложение матрицы в произведение  A = LU
    # U - верхнетреугольная, L - нижнетреугольная
    N = len(A)
    L = np.zeros(shape = (N, N))
    U = A
    for i in range(N):
        for j in range(i, N):
            L[j][i] = U[j][i] * 1.0 / U[i][i]

    for k in range(1, N):
        for i in range(k-1, N):
            for j in range(i, N):
                L[j][i] = U[j][i] * 1.0 / U[i][i]
        for i in range(k, N):
            for j in range(k-1, N):
                U[i][j] = U[i][j] - L[i][k-1] * U[k-1][j]
    return (L, U)

def solveU(U : list, F : list) -> list:
    # Решает СЛАУ UX = F
    # U - верхнетреугольная матрица
    N = len(U)
    Solution = np.zeros(N)
    Solution[N - 1] = F[N - 1, 0] * 1.0 / U[N - 1][N - 1]
    for n in range(N - 2, -1, -1):
        Sum = 0
        for i in range(n + 1, N):
            Sum += U[n][i] * Solution[i]
        Solution[n] = (F[n, 0] - Sum) / U[n][n]
    return Solution

def getSolution(Matrix : list, F : list) -> list:
    # Решает СЛАУ
    N = len(Matrix)
    L, U = LUdecomposition(Matrix)
    F = np.dot(np.linalg.inv(L), F)
    return solveU(U, F)

def getFunctionValues(Kfunc : "function", g : "function", f : "function", Lambda : float, a : float, b : float, N : int, ArrayX0 : list) -> (list, list):
    # Возвращает 
    # Nodes    - массив из N + k точек xk, полученных равным делением отрезка [a, b] на N - 1 частей и дополненный k точками
    # FunctionValues - массив значений функции u(x) в этих точках
    
    # 1. Делим отрезок на N-1 частей
    Nodes = np.linspace(a, b, N)
    
    # 2. Дополняем его так, чтобы масимальное из X0 попадало внутрь интервала интерполяции
    h = (b - a) / (N - 1)
    K = int((max(ArrayX0) - b) / h) + 1
    AddNodes = [b + i * h for i in range(1, K + 1)]
    Nodes += AddNodes
    
    # 3. Ищем веса Weights квадратурной формулы, интегрируя базовые многочлены Лагранжа
    Weights = []
    for k in range(1, N + 1):
        # Интегрирование методом трапеций по 1000 точкам
        Args = np.arange(a, b, 0.001)
        BaseLagranValues = [problem_3.get_l(i, k - 1, Nodes) for i in Args]
        Weights.append(problem_4.calculateSimpson(BaseLagranValues, a, b))
        
    # 4. Получаем матрицу СЛАУ Matrix и столбец правых частей F
    Matrix = np.zeros(shape = (N + K, N + K))
    for i in range(N + K):
        for j in range(N + K):
            if (j < N):
                Matrix[i, j] = -Lambda * Weights[j] * Kfunc(Nodes[i], Nodes[j])
    for i in range(N + K):
        Matrix[i][i] += g(Nodes[i])
        
    F = np.zeros(shape = (N + K, 1))
    for i in range(N + K):
        F[i, 0] = f(Nodes[i])
        
    # 5. Решаем СЛАУ
    FunctionValues = getSolution(Matrix, F)
    
    return Nodes, FunctionValues

def getFunctionValue(X0 : float, ArgValues : list, FunctionValues : list) -> float:
    NumArg = 0
    while (ArgValues[NumArg] < X0):
        NumArg += 1
    return FunctionValues[NumArg]
    
        
def main():
    a = -1
    b = 1
    Lambda = -1
    # Число узлов
    ArrayN = [3, 4, 5, 6]
    # Точки, в которых хотим получить значение u(x)
    ArrayX0 = [1.1, 1.25, 1.5]
    
    plt.figure(figsize = (10, 10))
    plt.title("Интерполяционный многочлен Ньютона для функции u(x)")
    
    for N in ArrayN:
        ArgValues, FunctionValues = getFunctionValues(K, g, f, Lambda, a, b, N, ArrayX0)
        plt.scatter(ArgValues, FunctionValues, marker = "^")
        Args = np.arange(-1, max(ArgValues) + 0.01, 0.01)
        NewtonVals = problem_3.getNewtonValues(Args, ArgValues, FunctionValues)

        print("\n\nЗначения в точках с помощью построения полинома по n =", N, "узлам:\n")
        for X0 in ArrayX0:
            print("X0 =", X0, ", u(X0) =", format(getFunctionValue(X0, Args, NewtonVals), '.4f'), "\n")
        
        NameGraph = "Ньютон n =" + str(N)
        plt.plot(Args, NewtonVals, label = NameGraph)

    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()