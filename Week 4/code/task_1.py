import numpy as np
import matplotlib.pyplot as plt

t0, tn = 0, 1
x0, xn = 0, 2
N = 100
delta_x = (xn - x0) / N

delta_t = 0.02
N_t = int((tn - t0) / delta_t)

a = 1
C = a * delta_t / delta_x

X = np.linspace(x0, xn, N + 1)
T = np.arange(t0, tn + delta_t, delta_t)


def f(x):
    return np.exp(-200 * (x - 0.2) ** 2)


U = np.zeros((N_t + 1, N + 1))
U[0, :] = f(X)

for i in range(1, N_t + 1):
    for j in range(1, N + 1):
        U[i, j] = U[i - 1, j] - C * (U[i - 1, j] - U[i - 1, j - 1])


plot_times = [0.25, 0.5, 0.75, 1.0]
plot_indices = [int(t / delta_t) for t in plot_times]
colors = ['blue', 'red', 'green', 'yellow']


plt.figure(figsize=(8, 5))
for i, idx in enumerate(plot_indices):
    plt.plot(X, U[idx], color=colors[i], label=f"t = {plot_times[i]}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("1D Advection Equation")
plt.legend()
plt.grid()
plt.show()
