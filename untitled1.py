import numpy as np
import matplotlib.pyplot as plt
import numpy.random as sim
from scipy.stats import norm

sig = 0.25
T = 10
rf = 0
n = 100
step = T/n
N = 100  # nombre de stocks
S = 50*np.ones((n+1, N))
K = 30
for j in range(N):
    for i in range(1, n+1):
        S[i, j] = S[i-1, j] + sig*S[i-1, j]*np.sqrt(step)*sim.randn()

dates = np.linspace(0, T, n+1)
plt.plot(dates, S)
plt.show()

G = norm()


def FI(x):
    return G.cdf(x)


def d(t, x):
    if T == t:
        return 0
    else:
        return np.log(x/K)/(np.sqrt(T-t)*sig) + 0.5*np.sqrt(T-t)*sig


def g(t, x):
    if t == T:
        y = max(x-K, 0)
    else:
        y = x*FI(d(t, x)) - K*FI(d(t, x) - sig*np.sqrt(T-t))
    return y


V0 = g(0, 50)
V = V0*np.ones((n+1, N))

for j in range(N):
    for i in range(1, n+1):
        V[i, j] = V[i-1, j] + FI(d(step*i, S[i, j]))*(S[i, j] - S[i-1, j])


plt.plot(dates, V)
plt.show()

gam = 0.2
a = 0.15
b = 0.03
r0 = 0.02
B = np.zeros((n+1, N))
r = r0*np.ones((n+1, N))
for i in range(1, n+1):
    for j in range(N):
        r[i, j] = r[i-1, j]+a*(b-r[i-1, j])*step+gam*np.sqrt(np.abs(r[i-1,j]))*np.sqrt(step)*sim.randn()

plt.plot(dates,r)
plt.show()