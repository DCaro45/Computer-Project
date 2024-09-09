"""Creating a symbolic analytical solution to the PDF of a particle in anharmonic potentials"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os
dir, file = os.path.split(__file__)


'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 4     # finish time
div = 1     # division of time points (i.e whole numbs, half, third etc))

epsilon = 10   # change in delta_xs size from spatial lattice spacing
bins = 100     # number of bins for histogram

N_cor = 20        # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor    # number of sweeps through path set
N_CF = 10 ** 4    # number of updates

'''determinants/shorthands'''
n_tp = int( div * (t_f - t_i) + 1 )         # number of temporal points
n_tl = int( div * (t_f - t_i) )             # number of temporal links
a = (t_f - t_i) / n_tl                  # size of time step
t_points = np.linspace(t_i, t_f, n_tp)  # temporal lattice points


m = mass           # shorthand for mass
nt = n_tp           # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' +
      'N_cor/Update = ' + str(N_cor) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N))


def pot1(x):
    V = 1/2 * x ** 2
    return V


def pot2(x):
    a = 1
    b = -1/12
    c = 1/1500
    d = 1/10000
    p = 1/8
    V = a * x + b * x ** 3 + c * x ** 5 + d * np.exp(x ** 2) ** p
    return V


def pot3(x):
    """ a polynomial potential with a minimum and a stationary inflection point"""
    V = - 3 * x ** 2 - 1/4 * x ** 3 + 1/2000 * x ** 6
    return V


def pot4(x):
    V = x ** 2
    return V


def pot5(x):
    if -5 < x < 5:
        V = x
    else:
        V = 1000000
    return V


def pot6(x):
    """a potential with the same form as the higgs potential"""
    u = 2
    l = 1
    V = - 0.5 * u ** 2 * x ** 2 + 0.25 * l ** 2 * x ** 4
    return V

def analytic(x, potential):
    """analytical solution to the PDF"""
    #KE = 1/2 * m * (x/a) ** 2
    PE = potential(x)
    #E_tot = KE + PE
    #Action = a * E_tot
    #prob = np.exp(-Action)
    prob = np.exp(-PE)

    return prob


def actn(x, j, potential):
    """calculating energies"""
    # setting index so that it loops around
    N = len(x)
    jp = (j-1) % N
    jn = (j+1) % N

    # calculating energies ... strange???
    KE = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    PE = potential(x[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    # keeping count of number of changes
    count = 0

    for j, x in enumerate(path):
        # creating a perturbed path from initial path
        dx = rdm.uniform(-e, e)

        eval_p = path.copy()
        eval_p[j] = x + dx

        # calculating actions
        S1 = actn(path, j, potential)
        S2 = actn(eval_p, j, potential)
        dS = S2 - S1

        # applying metropolis logic
        r = rdm.random()
        W = np.exp(-dS)
        if dS < 0 or W > r:
            path = eval_p
            count += 1

    return path, count


pot = pot6


"""Initialising paths and trialing metropolis"""

p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)


"""Thermalising lattice"""

init = p_1
'''
for i in range(T):
    new_p, counts = Metropolis(init, pot)
    init = new_p
'''

"""Applying metropolis"""
all_ps = [init]
t_counts = 0
# applying metropolis to path N_CF times
for j in range(N_CF):
    # initialising starting path
    start_p = all_ps[-1]
    # applying metropolis to path N_cor times
    for i in range(U):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts

    # adding final path to all_ps
    all_ps.append(start_p)

print('prop of changing point = ' + str(t_counts/(nt*U*N_CF)))

"""all points fromn new_ps"""
ln = len(all_ps)
pos = np.zeros([ln * nt])
k = 0
for i in range(ln):
    for j in range(nt):
        pos[k] = all_ps[i][j]
        k += 1


xs = np.linspace(min(pos) - 5, max(pos) + 5, 1000)
y = np.zeros([len(xs)])
V = np.zeros([len(xs)])
for i, x in enumerate(xs):
    y[i] = analytic(x, pot)
    V[i] = pot(x)

Norm, bins = np.histogram(pos, bins=bins, density=True)
y = y / max(y) * max(Norm)

fig, ax1 = plt.subplots()
ax1.stairs(Norm, bins, fill=True, label='Monte Carlo integral')
ax1.tick_params(axis='y', labelcolor='tab:orange')
ax1.plot(xs, y, color='tab:orange', label='Analytic solution')
ax1.set_xlabel('Position')
ax1.set_ylabel('Probability Density', color='black')
plt.legend()
ax2 = ax1.twinx()
ax2.plot(xs, V, color='red', label='Potential')
ax2.tick_params(axis='y', labelcolor='red')
ax2.set_ylabel('Potential', color='black')
plt.legend(loc='upper left')
fig.tight_layout()
fig.savefig(dir + '\\Images\\calc_analytical_solution_6.png')
plt.show()

