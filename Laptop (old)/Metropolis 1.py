"""operating at one point in each path, keeping last set of paths"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 7                         #Number of time points
divs_x = 5                      #Division of space points (i.e whole numbs, half, third etc)
N= 10e+2                        #number of samples for prop calc

_size_ = 5                    #how much smaller the delta_xs shouls be from lattice spacing
Sweeps = 10e+4
bins = 40


'''determinants/shorthands'''

N = int(N)
N2 = int(Sweeps)

n_x = divs_x * (x1 - x0) + 1         #number of spatial points
a = (tf-ti)/(n_t-1)                  #size of time step
b = (x1-x0)/(n_x-1)                  #size of spacial step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

m = mass                         #shorthand for mass
n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points



def pot(x):
    """simple harmonic oscillator potential"""

    V = 1/2*(x)**2
    return V

def path_gen1(xs, N):
    """path generator"""
    #n is the length of the spatial lattive array
    N = int(N)
    div = int(N/n)
    paths = np.zeros([N, n])
    indx = 0
    for x0 in xs:
        for m in range(div):
            for i in range(n):
                paths[indx][i] = x0
            indx +=1
    return paths

def actn(path, potential):
    """calculating energies"""

    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    Action = a * E_path
    return Action

def wght(action):
    """calculating weight"""
    weight = np.exp(-action)
    return weight

def pdf(x):
    """prob density function"""

    prob = (np.exp(-(x ** 2 / 2)) / np.pi ** (1 / 4)) ** 2
    return prob

def norm(array):
    """normalisation function"""

    total = sum(array)
    if total > 0:
        normalised = array/total
        return normalised
    else:
        return 0

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    epsilon = b/_size_
    e = epsilon
    dx = rdm.uniform(-e, e)
    indx = rdm.randrange(0, n)

    #old E
    old_p, new_p = path, path
    S_old = actn(old_p, potential)

    #new E
    new_p = np.zeros([len(path)])
    for i, x in enumerate(path):
        if i == indx:
            new_p[i] = x + dx
        else:
            new_p[i] = x
    S_new = actn(new_p, potential)

    #delta S
    dS = S_new - S_old

    #conditional statement
    #cont = []
    eval_p = []
    'make a dS cut off'
    if dS < 0:
        eval_p = new_p
        #cont = True
    elif dS > 0:
        r = rdm.random()
        W = wght(dS)
        if W > r:
            eval_p = new_p
            #cont = True
        else:
            eval_p = old_p
            #cont = False
    return eval_p #, cont

print(n, N, N2)


p_1 = path_gen1(x, 5)
#print(p_1[0])
new_p, count = Metropolis(p_1[0], pot)
print(new_p, count, (count/n))
old_ps = path_gen1(x, N)


"""generating paths and appluiong metropolis"""

old_ps = path_gen2(x, N)
new_ps = old_ps
j = 0
for i in range(N2):
    for j in range(N):
        new_ps[j] = Metropolis(old_ps[j], pot)
#print(new_ps)

pos = np.zeros([N * n])
m = 0
for i in range(N):
    for j in range(n):
        pos[m] = new_ps[i][j]
        m += 1


xs = np.linspace(min(pos), max(pos), len(pos))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)

plt.hist(pos, bins=bins)
plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.show()

