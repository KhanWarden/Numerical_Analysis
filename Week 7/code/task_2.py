import numpy as np
import matplotlib.pyplot as plt


def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_ = np.zeros(n - 1)
    d_ = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n - 1):
        c_[i] = c[i] / (b[i] - a[i - 1] * c_[i - 1])
    for i in range(1, n):
        d_[i] = (d[i] - a[i - 1] * d_[i - 1]) / (b[i] - a[i - 1] * c_[i - 1])

    u = np.zeros(n)
    u[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        u[i] = d_[i] - c_[i] * u[i + 1]

    return u


L = 1.0
N = 10
h = L / (N + 1)
A, B = 0, 1


def f(x):
    return 6 * x


x = np.linspace(h, L - h, N)
b = -2 * np.ones(N)
a = np.ones(N - 1)
c = np.ones(N - 1)
d = -h ** 2 * f(x)

d[0] -= A
d[-1] -= B

u = thomas_algorithm(a, b, c, d)

x_full = np.linspace(0, L, N + 2)
u_full = np.concatenate(([A], u, [B]))

plt.plot(x_full, u_full, 'o-', label="Численное решение")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.legend()
plt.grid()
plt.show()
