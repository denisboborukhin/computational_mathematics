import numpy as np
import scipy as sp
import matplotlib as mpl
from numpy import float32, float64
from numpy import linalg
import matplotlib.pyplot as plt
from functools import partial
import itertools

def newton_diff_order_n(xs: np.array, ys: np.array) -> float64:
    n = len(xs)
    if n == 1:
        return ys[0]

    elif n == 2:
        x1, x2 = xs
        y1, y2 = ys

        return (y2 - y1) / (x2 - x1)

    x0 = xs[0]
    xn = xs[-1]

    return (newton_diff_order_n(xs[1:], ys[1:]) - newton_diff_order_n(xs[0:-1], ys[0:-1])) / (xn - x0)

    
def newton_interpolation(xs: np.array, ys: np.array) -> np.array:
    current_poly = np.array([1.0])
    result_poly = np.array([0.0])

    for n, x in enumerate(xs):
        result_poly = np.polyadd(result_poly, newton_diff_order_n(xs[0:n + 1], ys[0:n + 1]) * current_poly)
        multiple_poly = np.array([1.0, -x])
        current_poly = np.polymul(current_poly, multiple_poly)
    
    return result_poly

def plot_points_and_poly(ax, xs: np.array, ys: np.array, poly: np.array, color: str = "blue"):
    num_points = 10000
    xs_linspace = np.linspace(min(xs), max(xs), num_points)

    ax.scatter(xs, ys, color=color)
    ax.plot(xs_linspace, np.polyval(poly, xs_linspace))

    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)

    ax.grid(True, which="both")
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    plt.show()

if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    xs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ys = np.array([1.0, 0.8, 0.5, 0.307, 0.2, 0.137, 0.1, 0.075, 0.06, 0.047, 0.039])
    poly_x_t = newton_interpolation(xs, ys)

    plot_points_and_poly(ax, xs, ys, poly_x_t)
    fig.savefig("output/5_1.pdf", transparent=False, bbox_inches="tight")