import numpy as np
import matplotlib.pyplot as plt

x0, xn = 0, 2
N = 100
delta_x = (xn - x0) / N

t0, tn = 0, 1
delta_t = 0.001
N_t = int((tn - t0) / delta_t)

nu = 0.1

X = np.linspace(x0, xn, N + 1)
T = np.arange(t0, tn + delta_t, delta_t)

U = np.zeros((N_t + 1, N + 1))
U[0, :] = np.sin(np.pi * X)

for i in range(1, N_t + 1):
    U_next = U[i - 1].copy()
    for j in range(1, N):
        convective = U[i - 1, j] * (U[i - 1, j] - U[i - 1, j - 1]) / delta_x
        diffusive = nu * (U[i - 1, j + 1] - 2 * U[i - 1, j] + U[i - 1, j - 1]) / delta_x**2
        U_next[j] = U[i - 1, j] - delta_t * convective + delta_t * diffusive
    U[i] = U_next.copy()

plot_times = [0.2, 0.4, 0.6, 0.8, 1.0]
plot_indices = [int(t / delta_t) for t in plot_times]
colors = ['blue', 'red', 'green', 'yellow', 'black']

plt.figure(figsize=(8, 5))
for i, idx in enumerate(plot_indices):
    plt.plot(X, U[idx], color=colors[i], label=f"t = {plot_times[i]:.1f}")

plt.xlabel("x")
plt.ylabel("u(x, t)")
plt.title("Burger's Solution")
plt.legend()
plt.grid()
plt.show()
