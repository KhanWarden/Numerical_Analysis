import numpy as np
import matplotlib.pyplot as plt

alpha = 0.2
beta = 0.1
t0, t_final = 0, 10
dt = 0.1
N0 = 50

t_values = np.arange(t0, t_final + dt, dt)
N_values = np.zeros_like(t_values)
N_values[0] = N0

for i in range(1, len(t_values)):
    N_values[i] = N_values[i-1] + dt * (alpha - beta) * N_values[i-1]

N_exact = N0 * np.exp((alpha - beta) * t_values)

plt.plot(t_values, N_values, '.', label="Euler")
plt.plot(t_values, N_exact, '-', label="Exact Solution")
plt.xlabel("Time (t)")
plt.ylabel("Population (N)")
plt.legend()
plt.grid()
plt.show()
