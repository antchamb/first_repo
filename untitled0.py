

import numpy as np
import numpy.random as sim # simulate r.v
import matplotlib.pyplot as plt # graphiques

T=3;n=1000; N=1000;

step=T/n
S0=100

K=70
r0=0.01
tildeS=S0*np.ones((n+1,N))
S=S0*np.ones((n+1,N))
r=r0*np.ones((n+1,N))
S0t=np.ones((n+1,N))
payoff=[]
payoff1=[]

a=0.5
b=0.02
gam=0.15

def Call(x,k):
    return max(x-k,0)

def sigma(t,x):
    return 0.1*(1+t/T+1/(1+x**2))

for j in range(N):
    for i in range(1,n+1):
        tildeS[i,j]=tildeS[i-1,j]+sigma(step*(i-1),tildeS[i-1,j])*tildeS[i-1,j]*np.sqrt(step)*sim.randn()
        r[i,j]=r[i-1,j]+a*(b-r[i-1,j])*step+gam*np.sqrt(step)*sim.randn()
        S0t[i,j]=S0t[i-1,j]+r[i,j]*S0t[i-1,j]*step
        S[i,j]=S0t[i,j]*tildeS[i,j]
    payoff.append(Call(S[n,j],K)/S0t[n,j])
    averagetraj=np.mean(S[:,j])
    payoff1.append(Call(averagetraj,K)/S0t[n,j])

dates=np.linspace(0,T,n+1)
plt.plot(dates,S0t)
plt.show()

V0=np.mean(payoff1)
print("V0=",V0)
