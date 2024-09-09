import numpy as np
import math as mt
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
n_x = divs_x * (x1 - x0) + 1    #number of spatial points

_size_ = 5                    #how much smaller the delta_xs shouls be from lattice spacing
N= int(10e+4)                   #number of samples for prop calc

'''determinants/shorthands'''
m = mass                       #shorthand for mass

a = (tf-ti)/(n_t-1)                  #size of time step
b = (x1-x0)/(n_x-1)                  #size of spacial step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points


def path_gen(xs, x0):
    """path generator"""

    lgt = len(xs)
    path = np.zeros([lgt])
    path[0]=path[lgt-1]=x0
    for i in range(1,lgt-1):
        path[i]=rdm.choice(xs)
    return path

def pot(x):
    """simple harmonic oscillator potential"""

    V = 1/2*(x)**2
    return V

def Actn(path, potential):
    """calculating energies"""

    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    Action = a * E_path
    return Action

def Wght(action):
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

def Metropolis(size, path, potential, action):
    """creating the metropolis algorithm"""
    dev = b/size
    dx = rdm.uniform(-dev, dev)
    point = int(rdm.choice(np.linspace(0, n-1, n-1)))

    #old E
    old_p, new_p = path, path
    E_old = action(old_p, potential)

    #new E
    new_p = np.zeros([len(path)])
    for i, x in enumerate(path):
        if i == point:
           new_p[i] = x + dx
        else:
            new_p[i] = x
    E_new = action(new_p, potential)

    #delta S
    dS = a * (E_new - E_old)

    #conditional statement
    cont = []
    eval_p = []
    'make a dS cut off'
    if dS < 0:
        eval_p = new_p
        cont = True
    elif dS > 0:
        r = rdm.random()
        W = Wght(dS)
        if W > r:
            eval_p = new_p
            cont = True
        else:
            eval_p = old_p
            cont = False
    return eval_p, cont

p_1 = path_gen(x, x[0])
new_p, C = Metropolis(5, p_1, pot, Actn)

"""initialising a set of paths"""
'just need to make the paths all either 0 or the same the whole way through'
xs = list(x)
run = int(N//n)
N2 = int(run*n)
Paths = np.zeros([N2,n])
for j, x0 in enumerate(xs):
    for i in range(run):
        Paths[i+j*run] = path_gen(xs,x0)

'run metropolis through these paths an x number of times'
'perform metropolis on each path the same number of times as the paths on the edges are optimised more so contribute a higher weighting than needed'
skip = 10
lgt = int(N/skip)
new_ps = np.zeros([lgt,n])
j = 0
for i in range(lgt):
    if j > N2:
        j = N2
        new_path, C = Metropolis(5, Paths[j], pot, Actn)
        new_ps[i] = new_path
        while C == True:
            new_path, C = Metropolis(10, new_path, pot, Actn)
            new_ps[i] = new_path
    else:
        new_path, C = Metropolis(5, Paths[j], pot, Actn)
        new_ps[i] = new_path
        while C == True:
            new_path, C = Metropolis(10, new_path, pot, Actn)
            new_ps[i] = new_path
    j += skip

def prop(points, paths, potential, action):
    "calculating propagator"

    G = np.zeros([n])
    for i in range(lgt):
        path = paths[i]
        S = action(path, potential)
        W = Wght(S)
        x0 = path[0]
        for x in points:
            if x-b < x0 < x+b:
                indx = np.where(points == x)
                G[indx] += W
    return G

Gs = prop(x, new_ps, pot, Actn)
pdf_A = pdf(x)

ys = norm(Gs)
y2 = norm(pdf_A)

plt.plot(x, ys)
plt.plot(x, y2)
plt.show()

