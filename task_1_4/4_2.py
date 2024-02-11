import numpy as np
import matplotlib.pyplot as plt
import math

def solve_iteration(x0, iteration_step, eps=1e-4):
    x = x0
    x_prev = [1000, 1000]
    i = 0
    error = 2 * eps
    
    while error > eps:
        x_prev = x.copy()
        x = iteration_step(x_prev)
        i += 1
        error = vec_norm([x[i] - x_prev[i] for i in range(len(x))])
    return x, i

def iterative_scheme(x):
    x_new = math.pow(((x[1] ** 2) - 1.98) / 2, 1/3)
    y_new = x[0] + 1.03 / x[0]
    return [x_new, y_new]

def system(x):
    return [x[0] * x[1] - x[0] ** 2 - 1.03, -2 * (x[0] ** 3) + x[1] ** 2 - 1.98]

def vec_norm(x):
    return math.sqrt(sum([i ** 2 for i in x]))

x0 = [1, 2]
res, iterations = solve_iteration(x0, iterative_scheme)
print("Result: {}" .format(res))
print("Number of iterations: {}" .format(iterations))