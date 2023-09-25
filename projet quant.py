import numpy as np
import numpy.random as sim
import matplotlib.pyplot as plt

X0 = 0.1 #paramètres en t=0
S0 = 20
T = 10 # on fixe un T
n = 1000 #nombre de part de découpage de l'intervalle [0,T]. On peut encore augmenter n si on veut une
#meilleut approximation de ce qui se passe en temps continue
W = np.zeros(n+1) #création de tableau/vecteur dans lesquels nous mettrons nos simulations de mouvements browniens
B = np.zeros(n+1)
X = X0 * np.ones(n+1) #cration d'un tableau/vecteur dans lequel nous mettrons notre simulation du processus X
step = T/n #

for i in range(1, n+1):
    W[i] = W[i-1] + np.sqrt(step) * sim.randn() #ici on simule des mouvements browniens et on remplit les tableaux
    B[i] = B[i-1] + np.sqrt(step) * sim.randn()


for i in range(1, n+1):
    X[i] = X[i-1] * (1 + 0.1 * (B[i] - B[i-1])) #comportement du processus X compte tenu de la simulation du mouvement brownien


dates = np.linspace(0, T, n+1) #création de l'axe de temps pour les graphiques. Notre unité temporelle est T/n !

plt.plot(dates, X)
plt.title("évolution du processus X")
plt.show()
#création de graphique pour mieux y voir

def max(x, y):
    if x > y:
        return x
    else:
        return y
#création d'une fonction max pour calculer sigma


sigma = np.ones(n+1) #création d'un vecteur de 1
for i in range(n+1):
    sigma[i] = 0.05 + max(X[i], 0.15)

plt.plot(dates, sigma)
plt.title("volatilité stochastique")
plt.show()
#☺graphique

r = 0.01 * np.ones(n+1)
for i in range(1, n+1):
    r[i] = r[i-1] + 0.1 * (0.03 - r[i-1])*step + 0.05 * (B[i] - B[i-1])#on remplit le vecteur r

plt.plot(dates, r)
plt.title("évolution du taux sans risque")
plt.show()

S_0 = np.ones(n+1) #actif sans risque
for i in range(1,n+1):
    S_0[i] = S_0[i-1] * (1 + r[i-1] * step) #on remplit

plt.plot(dates, S_0)
plt.title("évolution de l'actif sans risque")
plt.show()

Stil0 = 20
Stil = Stil0 * np.ones(n+1) #actif risqué actualisé à l'actif sans risque
for i in range(1,n+1):
    Stil[i] = Stil[i-1] * (1 + sigma[i-1] * (W[i] - W[i-1]))#on remplit

plt.plot(dates, Stil)
plt.title("évolution du processus Stil")
plt.show()

S = Stil0 * np.ones(n+1)
for i in range(1,n+1):
    S[i] = Stil[i] * S_0[i]

plt.plot(dates, S)
plt.title("évolution de l'actif risqué")
plt.show()

K1 = 10
K2 = 40
#valeurdonnées
L = []
S0_T = []
#création de liste pour garder les données que l'on veut, L = les KSI1(j)_T
L3 = [] #pour les KSI2(j)_T
N = 1000 #ici on suppose que Nest très grand et qu'on étudit le comlportement à la limite.
#on pourrait augmenter N pour plus de précision mais c'est en fonction de l'ordinateur

for j in range(1,N):
    L2 = [] #pour le calcul du gains des options asiatiques, on prend dans L2 toutes les valeurs de la simulation
    #comme on est en temps discret l'intégrale devient une somme des valeurs prises sur sur chaque intervalles
    #qui sont de même distance (mesure lebesgues(t+1 ; t) = step)
    W2 = np.zeros(n+1)
    B2= np.zeros(n+1)
    X2 = X0 * np.ones(n+1)
    sigma2 = np.ones(n+1)
    r2 = 0.01 * np.ones(n+1)
    Stil2 = Stil0 * np.ones(n+1)
    S_02 = np.ones(n+1)
    S2 = Stil0 * np.ones(n+1)
    #on refait comme avant pour la simulation j
    for i in range(1,n+1):
        W2[i] = W2[i-1] + np.sqrt(step) * sim.randn()
        B2[i] = B2[i-1] + np.sqrt(step) * sim.randn()
    for i in range(1, n+1):
        X2[i] = X2[i-1] * (1 + 0.1 * (B2[i] - B2[i-1]))
    for i in range(n+1):
        sigma2[i] = 0.05 + max(X2[i], 0.15)
    for i in range(1, n+1):
        r2[i] = r2[i-1] + 0.1 * (0.03 - r2[i-1])*step + 0.05 * (B2[i] - B2[i-1])
    for i in range(1,n+1):
        S_02[i] = S_02[i-1] * (1 + r2[i-1] * step)
    for i in range(1,n+1):
        S2[i] = Stil2[i] * S_02[i]
        L2.append(step*S2[i])#on collecte toute les valeurs prises par S2 sur chaque intervalle de temps
    L3.append(max((K2 - sum(L2)/T)/S_02[n],0)) #intégrale se transforme en somme, on est en temps discret.
    #on rajoute dans L3 le KSI2_T de chaque réalisation j
    L.append(max(((S2[n] - K1)/S_02[n]),0))
    #on rajoute dans L le KSI1_T de chaque réalisation j


print("évaluation du prix de l'option européenne: ",np.mean(L))
print("évaluation du prix de l'option asiatique: ", np.mean(L3))
