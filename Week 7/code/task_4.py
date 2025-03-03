import numpy as np
import matplotlib.pyplot as plt


L = 2.0
T = 1.0
N = 200
M = 400
dx = L / (N - 1)
dt = T / M
a = 1.0

CFL = a * dt / dx

x = np.linspace(0, L, N)
u = np.exp(-100 * (x - 0.5) ** 2)

time_steps = [0.25, 0.5, 0.75, 1.0]
solutions = []

for n in range(M):
    u_new = np.zeros_like(u)
    u_new[1:] = u[1:] - CFL * (u[1:] - u[:-1])

    u = u_new.copy()

    if (n + 1) * dt in time_steps:
        solutions.append(u.copy())

plt.figure(figsize=(8, 5))
colors = ['b', 'r', 'g', 'y']
for i, t in enumerate(time_steps):
    plt.plot(x, solutions[i], color=colors[i], label=f't = {t}')

plt.xlabel("x")
plt.ylabel("y")
plt.title("1D Advection Equation")
plt.legend()
plt.grid()
plt.show()
