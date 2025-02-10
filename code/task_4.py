import numpy as np
import matplotlib.pyplot as plt

alpha = 0.1
beta = 0.02
delta = 0.01
gamma = 0.1

X0 = 40
Y0 = 9
T = 200
dt = 0.1
N = int(T / dt)

time = np.linspace(0, T, N+1)
X = np.zeros(N+1)
Y = np.zeros(N+1)

X[0] = X0
Y[0] = Y0

for i in range(N):
    X[i+1] = X[i] + dt * (alpha * X[i] - beta * X[i] * Y[i])
    Y[i+1] = Y[i] + dt * (delta * X[i] * Y[i] - gamma * Y[i])

plt.plot(time, X, label="Prey Population", color='blue')
plt.plot(time, Y, label="Predator Population", color='red')
plt.xlabel("Time")
plt.ylabel("Population")
plt.legend()
plt.grid()
plt.show()
