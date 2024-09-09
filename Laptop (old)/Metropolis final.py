"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 4     # finish time
div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 0.5  # change in delta_xs size from spatial lattice spacing
bins = 100     # number of bins for histogram

N_cor = 10            # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5        # number of updates

'''determinants/shorthands'''
n_tp = div_t * (tf - ti) + 1          # number of temporal points
n_tl = div_t * (tf - ti)              # number of temporal links
a = (tf - ti) / n_tl                  # size of time step
t_points = np.linspace(ti, tf, n_tp)  # temporal lattice points

Sweeps1 = 10 * N_cor  # number of sweeps through path set
Sweeps2 = N_cor       # not necessary right now

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
S1 = int(Sweeps1)  # shorthand for sweeps 1 (and integer data type)
S2 = int(Sweeps2)  # shorthand for sweeps 2 (and integer data type)

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + 'N_cor = ' + str(N_cor) + ', ' + 'S1 = ' + str(S1) + ', ' + 'N_CF = ' + str(N_CF))

def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V

def actn(x, i, potential):
    """calculating energies"""
    jp = (i-1)%nt
    jn = (i+1)%nt

    KE = 0.5 * m * ((jp - x) ** 2 + (jn - x) ** 2 ) * (1/a) ** 2
    PE = potential(x)
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    eval_p = path
    count = 0

    for j,x in enumerate(eval_p):
        dx = rdm.uniform(-e, e)
        xP = x + dx

        S1 = actn(x, j, potential)
        S2 = actn(xP, j, potential)
        dS = S2 - S1

        r = rdm.random()
        W = np.exp(-dS)
        if dS < 0:
            eval_p[j] = xP
            count += 1
        elif W > r:
            eval_p[j] = xP
            count += 1
        else:
            eval_p[j] = x

    return eval_p, count


def pdf(x):
    """prob density function"""
    prob = (np.exp(-(x ** 2 / 2)) / np.pi ** (1 / 4)) ** 2
    return prob


def norm(array):
    """normalisation function"""

    total = sum(array)
    if total > 0:
        normalised = array / total
        return normalised
    else:
        return 0


p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)

'''

"""Thermalising lattice"""
init = p_1
array = [init]
for i in range(S1):
    new_p, counts = Metropolis(array[-1], pot)
    array.append(new_p)

"""generating paths and applying metropolis"""
all_ps = []
t_counts = 0
for j in range(N_CF):
    start_p = array[-1]
    for i in range(S2):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts
    all_ps.append(start_p)

#print(all_ps)
print('prop of changing point = ' + str(t_counts/(nt*N_CF*S2)))


"""points from new_ps skipping every N_cor'th one"""

pos_1 = np.zeros([int(len(all_ps)/N_cor) * nt])
print(N_cor, len(all_ps), len(all_ps) * nt, len(pos_1))
m = 0
for i in range(int(len(all_ps)/N_cor)):
    for j in range(nt):
        pos_1[m] = all_ps[int(i*N_cor)][j]
        m += 1
#print(pos_1)


"""all points fromn new_ps"""
ln = len(all_ps)
pos_2 = np.zeros([ln * nt])
m = 0
for i in range(ln):
    for j in range(nt):
        pos_2[m] = all_ps[i][j]
        m += 1
#print(pos_2)

xs = np.linspace(min(pos_1), max(pos_1), len(pos_1))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos_1, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)

plt.hist(all_ps, bins=bins, label ='M4', histtype="step")
#plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.legend()
plt.show()

xs = np.linspace(min(pos_2), max(pos_2), len(pos_2))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos_2, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)


plt.hist(pos_2, bins=bins, label = 'M4')
plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.legend()
plt.show()

'''