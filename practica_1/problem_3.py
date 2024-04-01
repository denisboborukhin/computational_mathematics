import numpy as np
import math
import matplotlib.pyplot as plt

def function_1(x : float) -> float:
    return 1.0 / (1 + 25 * x * x)

def get_l(t : float, k : int, ArgValues : list) -> float:
    n = len(ArgValues)
    lk = 1
    for j in range(n):
        Denom = ArgValues[k] - ArgValues[j]
        lk *= ((t - ArgValues[j]) / Denom) if k != j else 1
    return lk

def get_lagrange_val(t : float, ArgValues : list, FunctionValues : list) -> float:
    n = len(ArgValues)
    Value = 0
    for k in range(n):
        Value += (get_l(t, k, ArgValues) * FunctionValues[k])
    return Value

if __name__ == '__main__':
    a = -1
    b = 1

    plt.figure(figsize = (10, 10))
    plt.title("Интерполяционный многочлен Лагранжа при разных n")

    Args = np.arange(-1, 1.01, 0.01)
    Vals = [function_1(Arg) for Arg in Args]
    plt.plot(Args, Vals, 'r', label = "График исходной функции")

    ArrayN = [4, 6, 10]

    for N in ArrayN:
        ArgValues = np.linspace(a, b, N)
        FunctionValues = []
        for Arg in ArgValues:
            FunctionValues.append(function_1(Arg))
        plt.scatter(ArgValues, FunctionValues)

        Values = []
        ValuesErrors = []
        for i, Arg in enumerate(Args):
            val = get_lagrange_val(Arg, ArgValues, FunctionValues)
            Values.append(val)
            ValuesErrors.append(abs(val - Vals[i]))


        nameGraph = "n =" + str(N)
        plt.plot(Args, Values, label = nameGraph)

    plt.grid()
    plt.legend()
    plt.show()

#######################################################################
##############################Part_2###################################
#######################################################################

def get_F(k : int, n : int, ArgValues : list, FunctionValues : list) -> float:
    if (k == n):
        return FunctionValues[0]
    F2 = get_F(k + 1, n, ArgValues, FunctionValues[1:])
    F1 = get_F(k, n - 1, ArgValues, FunctionValues[:-1])
    t2 = ArgValues[n]
    t1 = ArgValues[k]
    DivDiff = (F2 - F1) / (t2 - t1)
    return DivDiff

def getNewton(ArgValues : list, FunctionValues : list) -> list:
    n = len(ArgValues)
    DivDiffs = []
    for k in range(n):
        F = get_F(0, k, ArgValues, FunctionValues)
        DivDiffs.append(F)
    return DivDiffs
    
def getNewtonValue(t : float, ArgValues : list, NewtonPol : list) -> float:
    Result = 0
    n = len(ArgValues)
    for k in range(n):
        Mult = 1
        for i in range(k):
            Mult *= (t - ArgValues[i])
        Result += Mult * NewtonPol[k]
    return Result

def getNewtonValues(Args : list, ArgValues : list, FunctionValues : list) -> list:
    NewtonValues = []
    NewtonPol = getNewton(ArgValues, FunctionValues)
    for Arg in Args:
        NewtonValues.append(getNewtonValue(Arg, ArgValues, NewtonPol))
    return NewtonValues

def getChebZeros(Start : float, Stop : float, N : int) -> float:
    Zeros = []
    HalfSum = (Start + Stop) / 2.0
    HalfDiff = (Stop - Start) / 2.0
    for ZeroNum in range(1, N + 1):
        Zero = HalfSum + HalfDiff * math.cos((2 * ZeroNum - 1) * math.pi / (2 * N))
        Zeros.append(Zero)
    return Zeros

if __name__ == '__main__':
    plt.figure(figsize = (10, 10))
    plt.title("Интерполяционный многочлен Ньютона с узлами в нулях полинома Чебышева")

    Vals = [function_1(Arg) for Arg in Args]
    plt.plot(Args, Vals, 'r', label = "График исходной функции")

    for N in ArrayN:
        NewtonArgValues = getChebZeros(a, b, N)
        FunctionValues = []
        for Arg in NewtonArgValues:
            FunctionValues.append(function_1(Arg))
        plt.scatter(NewtonArgValues, FunctionValues)

        NewtonValues = []
        NewtonErrors = []
        NewtonPol = getNewton(NewtonArgValues, FunctionValues)
        for i, Arg in enumerate(Args):
            val = getNewtonValue(Arg, NewtonArgValues, NewtonPol)
            NewtonValues.append(val)
            NewtonErrors.append(abs(val - Vals[i]))


        NameGraph = "n =" + str(N)
        plt.plot(Args, NewtonValues, label = NameGraph)

    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize = (10, 10))
    plt.title("Сравнение Лагранжa и Ньютона, n = 10")

    plt.plot(Args, NewtonErrors, label = "Newton Errors")
    plt.plot(Args, ValuesErrors, label = "Lagrange Errors")

    plt.grid()
    plt.legend()
    plt.show()
