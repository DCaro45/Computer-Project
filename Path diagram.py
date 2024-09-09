"""showing how the path is creating for brute force and metropolis algorithms"""

import numpy as np
import matplotlib.pyplot as plt
import random as rdm
import os

plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

dir, file = os.path.split(__file__)

'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 4     # finish time
x1 = 2      #upper bound
x0 = -x1    #lower bound
step = 0.5     # division of time points (i.e whole numbs, half, third etc))
x_step = 1

epsilon = 1   # change in delta_xs size from spatial lattice spacing

N_cor = 20       # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor    # number of sweeps through path set

save = False

'''determinants/shorthands'''
t_points = np.arange(t_i, t_f + step, step)  # number of temporal points
x_points = np.arange(x0, x1 + step, x_step)  #spacial lattice points
n_tp = len(t_points)          # number of temporal points
n_tl = n_tp - 1               # temporal lattice points


m = mass           # shorthand for mass
nt = n_tp           # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
x = x_points       # shorthand for spatial lattice points
e = epsilon        # shorthand for epsilon
a = step           # shorthand for size of time step
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'x = ' + str(x) + ', ' +
      'N_cor/Update = ' + str(U) + ', ' + 'Therm = ' + str(T))

def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V


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


def path_gen(xs, x0):
    path = np.zeros([nt])
    path[0] = path[nt-1] = x0
    for i in range(1,nt-1):
        path[i]=rdm.choice(xs)
    return path


"""Initialising paths and trialing metropolis"""

p_1 = [0 for X in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)


"""Metropolis Paths"""

init = p_1

fig = plt.figure(figsize=[6,6])
fig.tight_layout()
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
plt.tick_params(bottom=True, top=False, left=True, right=False)
plt.axvline(0, linestyle='dashed', color='black', linewidth='1', label='Origin')
for i in range(T):
    new_p, counts = Metropolis(init, pot)
    init = new_p
    if i == 0:
        plt.plot(init, t_points, label='Path')
    plt.plot(init, t_points, alpha=(0.98)**i)
plt.grid(axis = 'y')
plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.legend(loc='upper left')
if save == True:
    fig.savefig(dir + '\\Images\\Metropolis paths.png')
plt.show()



"""Brute force paths"""

samples = 5

plt.figure(figsize = [3, 4])
for i in range(0,samples):
    p = path_gen(x, x[np.where(x == 0)])
    plt.plot(p, t)
plt.grid(axis = 'x')
plt.show()