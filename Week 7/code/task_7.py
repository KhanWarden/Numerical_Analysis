import numpy as np
import matplotlib.pyplot as plt


k = 0.05


def reaction_system(t, u):
    CH4, O2, CO2, H2O = u
    rate = k * CH4 * O2
    dCH4_dt = - rate
    dO2_dt  = - 2 * rate
    dCO2_dt = rate
    dH2O_dt = 2 * rate
    return np.array([dCH4_dt, dO2_dt, dCO2_dt, dH2O_dt])


def euler_system(f, u0, t):
    n = len(t)
    dim = len(u0)
    U = np.zeros((n, dim))
    U[0] = u0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        U[i+1] = U[i] + h * f(t[i], U[i])
    return U


def rk4_system(f, u0, t):
    n = len(t)
    dim = len(u0)
    U = np.zeros((n, dim))
    U[0] = u0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], U[i])
        k2 = f(t[i] + h/2, U[i] + h*k1/2)
        k3 = f(t[i] + h/2, U[i] + h*k2/2)
        k4 = f(t[i] + h,   U[i] + h*k3)
        U[i+1] = U[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    return U

u0 = np.array([1.0, 0.1, 0.0, 0.0])

t = np.linspace(0, 5, 100)

sol_euler = euler_system(reaction_system, u0, t)
sol_rk4   = rk4_system(reaction_system, u0, t)

species = ['CH4', 'O2', 'CO2', 'H2O']
colors = ['blue', 'red', 'green', 'magenta']

plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.plot(t, sol_euler[:, i], 'o--', color=colors[i], label=f'{species[i]} (Euler)')
    plt.plot(t, sol_rk4[:, i], 's-', color=colors[i], label=f'{species[i]} (RK4)')
    plt.xlabel('Time')
    plt.ylabel(f'{species[i]} concentration')
    plt.title(f'{species[i]} vs Time')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()
