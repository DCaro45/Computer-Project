import numpy as np
import matplotlib.pyplot as plt
import math as mt
import random as rdm

'''values'''
mass = 1

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 7                         #Number of time points
divs_x = 2                      #Division of space points (i.e whole numbs, half, third etc)
n_x = divs_x * (x1 - x0) + 1    #number of spatial points

'sample sizes'
N= int(10e+1)                   #number of samples for prop calc

N_fin = 10e+1                   #finishing point logarithmic scale
base = 2                        #base for logarithmic scale
nbr = 10                        #number of graphs


'''determinants/shorthands'''
m = mass                       #shorthand for mass

a = (tf-ti)/(n_t-1)                  #size of time step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

n = n_x                          #shorthand for no.x points
x = x_points                     #shorthand for lattice points

exp = mt.log(N_fin, base)       #exponent for end value of logarithm range


"""Defining functions"""
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

def Engy(path, potential):
    """calculating energies"""

    E_path = [0]
    for i in range(0, n-1):
        KE = m/2*((path[i+1]-path[i])/a)**2
        PE = potential((path[i+1]+path[i])/2)
        E_tot = KE + PE
        E_path+=  E_tot
    return E_path

def Wght(energy):
    """calculating weight"""

    weight = np.exp(-a*energy)
    return weight

def prop(points, potential, path, energy, samples):
    """calculating propagator"""

    l_size = len(points)
    G = np.zeros([l_size])
    run = int(samples/l_size)
    for x0 in points:
        for i in range(0, run):
            p = path(points, x0)
            E = energy(p, potential)
            W = Wght(E)
            indx = np.where(points == x0)
            G[indx] += W
    return G

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


'''Plotting PDF'''
#values
G = prop(x, pot, path_gen, Engy, N)
Norm_G = norm(G)
y1 = Norm_G

#Graph
plt.figure()
plt.plot(x, y1)
plt.show()


'''repeating propagator for smaller samples'''
#Sample size
Ns = np.logspace(start=0, stop= exp, base=base, num=nbr)
lng = len(Ns)

#values
Gs = np.zeros([lng, n])
for j in range(0, lng):
    for i in Ns:
        Gs[j] = prop(x, pot, path_gen, Engy, int(i))

#normalising valuesa
Norm_Gs = np.zeros([lng,n])
for i in range(0,lng):
    Norm_Gs[i] = norm(Gs[i])
ys = Norm_Gs

'plotting graphs'
As = np.linspace(int(1/lng), 1, lng)    #Alpha valeus

plt.figure()
for j in range(0, lng):
    plt.plot(x, ys[j], alpha = As[j])
plt.show()


'''plot of FPI and standard formulation'''
#calculate potential and analytic pdf'
pdf_A = pdf(x)
y2 = norm(pdf_A)

l = 100 * (x1 - x0) + 1
xs = np.linspace(-2, 2, l)
ys = pot(xs)

#plotting graphs'
plt.figure()
plt.plot(x , y1, label = 'FPI', color = 'k')
plt.plot(x , y2, label = 'PDF', color = 'tab:orange' )
plt.plot(xs, ys, label = 'Potential', color = 'tab:blue')
plt.legend()
plt.grid()
plt.xlim(-2, 2)
plt.xlabel('position')
plt.ylabel('probability')
plt.ylim(0, max(y1) + 0.1*max(y1))
plt.show()





