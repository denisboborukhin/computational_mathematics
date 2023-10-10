import numpy as np
import matplotlib.pyplot as plt

def create_matrix(n, a):
    arr = []
    for i in range(n):
        arr.append(n * [0])
    
    for i in range(n):
        arr[i][i] = 2
    for i in range(n - 1):
        arr[i][i + 1] = - 1 - a
    for i in range(1, n):
        arr[i][i - 1] = - 1 + a

    arr = np.array(arr)
    return arr

a = 0
f = n * [0]
f[0] = 1 - a
f[n - 1] = 1 + a