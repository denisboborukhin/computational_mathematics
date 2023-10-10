import numpy as np
import math
import matplotlib.pyplot as plt

def f1(x, y):
    return x**2 + y**2 - 1

def f2(x, y):
    return math.tan(x) - y

def dichotomy_method(f1, f2, a, b, eps):
    roots = []
    for x in np.arange(a, b, 1):
        for y in np.arange(a, b, 1):
            x1, x2 = x, x + 1
            y1, y2 = y, y + 1

            if f1(x1, y1) * f1(x1, y2) < 0:
                root = dichotomy(f1, x1, x2, y1, y2, eps)
                if root is not None:
                    if abs(f2(root[0], root[1])) < eps:
                        roots.append(root)
            if f2(x1, y1) * f2(x2, y1) < 0:
                root = dichotomy(f2, x1, x2, y1, y2, eps)
                if root is not None:
                    if abs(f1(root[0], root[1])) < eps:
                        roots.append(root)

    return roots

def dichotomy(f, x1, x2, y1, y2, eps):
    while abs(x2 - x1) > eps:
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        if abs(f(x1, y1) * f(x, y)) < 0:
            x2, y2 = x, y
        else:
            x1, y1 = x, y
        
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    if abs(f(x, y)) > eps:
        return None

    return x, y

eps = 1e-6
roots = dichotomy_method(f1, f2, -10, 10, eps)
print("The roots of the system of functions are:", roots)

x = [root[0] for root in roots]
y = [root[1] for root in roots]
plt.scatter(x, y)
plt.show()