import numpy as np
import matplotlib.pyplot as plt

def get_l_k(xs: np.array, k: int) -> np.array:
    poly = np.array([1.0])
    div = 1.0
    
    for n, x in enumerate(xs):
        if n == k:
            continue
            
        poly = np.polymul(poly, np.array([1.0, -x]))
        div *= (xs[k] - xs[n])

    poly /= div
    return poly

def get_langrange_poly(xs: np.array, ys: np.array) -> np.array:
    result = np.array([0.0])
    for n, y in enumerate(ys):
        result = np.polyadd(result, get_l_k(xs, n) * y) 
    return result

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

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

xs = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ys = np.array([0.02, 0.079, 0.175, 0.303, 0.459, 0.638, 0.831, 1.03, 1.23, 1.42])
poly_y_t = get_langrange_poly(xs, ys)

plot_points_and_poly(ax, xs, ys, poly_y_t)
fig.savefig("output/5_2.pdf", transparent=False, bbox_inches="tight")