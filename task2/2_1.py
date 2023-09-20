########################Task_2_1###############################
# Answer to the problem: n = 3, at n = 5, the error increases,
# and at n = 7, it becomes about 10^6 
###############################################################

from scipy.misc import derivative

import numpy as np
import math
import matplotlib.pyplot as plt

def f(t):
    return math.sin(t)

def u_n_delta(n: int, t = 0.5):
    def u_diff_norm(n: int):    
        d_pts = n * 2
        if d_pts % 2 == 0:
            d_pts += 1
        return derivative(f, 0, 1e-5, n, order=d_pts) / math.factorial(n)

    res = 0
    for i in range (0, n + 1):
        res += u_diff_norm(i) * (t ** i)

    return math.fabs(res - f(t))

u_err = 1e-3
num_pts = 5

x = np.arange(1, num_pts + 1)
y = np.empty(num_pts)

min_n = -1
for i in range (1, num_pts):
    res = u_n_delta(i)
    y[i - 1] = res
    if (res < u_err and min_n == -1):
        min_n = i

print("Minimum n is ", min_n)

plt.plot(x, y)
plt.xlabel('n')
plt.ylabel('mistake')
plt.grid()
plt.show()
