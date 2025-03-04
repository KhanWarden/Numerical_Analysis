import numpy as np
import matplotlib.pyplot as plt


def thomas(a, b, c, d):
    n = len(d)
    cp = c.copy()
    dp = d.copy()

    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * cp[i - 1]
        dp[i] -= m * dp[i - 1]

    x = np.zeros(n)
    x[-1] = dp[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / b[i]
    return x


N = 50
dx = 1.0 / N
x = np.linspace(0, 1, N + 1)

k = 1.0
dt = 0.001
t_max = 0.1
n_steps = int(t_max / dt)
r = k * dt / dx ** 2

u_left = 0.0
u_right = 0.0

u = np.sin(np.pi * x)

for n in range(n_steps):
    n_int = N - 1
    a_arr = -r * np.ones(n_int)
    b_arr = (1 + 2 * r) * np.ones(n_int)
    c_arr = -r * np.ones(n_int)
    d_arr = u[1:-1].copy()
    d_arr[0] += r * u_left
    d_arr[-1] += r * u_right
    u_new_inner = thomas(a_arr.copy(), b_arr.copy(), c_arr.copy(), d_arr)
    u[0] = u_left
    u[-1] = u_right
    u[1:-1] = u_new_inner

plt.figure(figsize=(8, 5))
plt.plot(x, u, 'o-', label='Распределение температуры при t_max')

plt.xlabel('x')
plt.ylabel('u(x, t_max)')
plt.title('1D уравнение теплопроводности (неявная схема, метод Томаса)')
plt.legend()
plt.grid()

plt.annotate("u(0)=0", xy=(0, u_left), xytext=(0.05, u_left + 0.2),
             arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)
plt.annotate("u(1)=0", xy=(1, u_right), xytext=(0.8, u_right + 0.2),
             arrowprops=dict(facecolor='black', arrowstyle="->"), fontsize=9)

plt.show()
