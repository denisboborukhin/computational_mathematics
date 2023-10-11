import numpy as np
import matplotlib.pyplot as plt
import math

def solve_newton(x0, max_i=1000, eps=1e-6):
    x = x0
    i = 0
    error = 2 * eps

    while (error > eps) and (i < max_i):
        fx = system(x)
        J = jacobi_matrix(x)
        dx = np.linalg.solve(J, fx)
        error = vec_norm(dx)
        x = x - dx
        i += 1

    return x

def system(x):
    return [x[0] ** 2 + x[1] ** 2 - 1, np.tan(x[0]) - x[1]]

def jacobi_matrix(x):
    J = [[2 * x[0], 2 * x[1]], [1 / np.cos(x[0]) ** 2, -1]]
    return J

def vec_norm(x):
    return math.sqrt(sum([i ** 2 for i in x]))

# Основное тело
x0 = [1, 1]
x1 = [-1, -1]

root_1 = solve_newton(x0)
root_2 = solve_newton(x1)

print("Root 1: {}".format(root_1))
print("Root 2: {}".format(root_2))

plt.scatter(root_1[0], root_1[1])
plt.scatter(root_2[0], root_2[1])
plt.title("Решение системы")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(which='major', color='k', linewidth=0.3)
plt.show()