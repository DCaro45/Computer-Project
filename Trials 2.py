"""operating at each point on path, keep each path iteration, start at the origin"""
#trial
#trials 2

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

'''values'''
mass = 1   # setting mass to be 1

ti = 0     # start time
tf = 4     # finish time
div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.3  # change in delta_xs size from spatial lattice spacing
N_cor = 25        # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 2    # number of updates

bins = 1000      # number of bins for histogram

'''determinants/shorthands'''
n_tp = div_t * (tf - ti) + 1          # number of temporal points
n_tl = div_t * (tf - ti)              # number of temporal links
a = (tf - ti) / n_tl                  # size of time step
t_points = np.linspace(ti, tf, n_tp)  # temporal lattice points

Therm = 5 * N_cor    # number of sweeps through path set
Update = N_cor        # not necessary right now

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
U = int(Update)    # shorthand for sweeps 2 (and integer data type)

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) +
      ', ' 'N_cor/Update = ' + str(N_cor) + ', ' + 'Thermal sweep = ' + str(T) + ', ' + 'N_CF = ' + str(N_CF))

def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V


def pot2(x):
    V = - 1/3 * x ** 2 + 1/200 * (x ** 5) + np.cos(x)
    return V

def pot3(x):
    """ a polynomial potential with a minimum and a stationary inflection point"""
    V = 1/2 * x ** 2 + 1/4 * x ** 4 - 1/20 * x ** 5
    return V
def pot4(x):
    V = - x ** 2
    return V

def pot5(x):
    if -5 < x < 5:
        V = x
    else:
        V = 100000
    return V

def pot6(x):
    """a potential with the same form as the higgs potential"""
    u = 2
    l = 1
    V = - 0.5 * u ** 2 * x ** 2 + 0.25 * l ** 2 * x ** 4
    return V

pot = pot5
xs = np.linspace(-5, 5, 100)
V = []
for x in xs:
    V.append(pot(x))

plt.plot(xs, V)
plt.xlim(-3, 3)
plt.ylim(0,4)
plt.show()

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


p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)


init = p_1
for i in range(T):
    x, count = Metropolis(init, pot)
    init = x
    plt.plot(x, t, alpha = 0.2)
therm = init
plt.xlabel('x')
plt.ylabel('t')
dir, file = os.path.split(__file__)
plt.savefig(dir + '\\Images\\MA_paths.png')
plt.show()

"""generating applying metropolis to inital path"""

init = p_1
all_ps = []
t_counts = 0
# applying metropolis to path N_CF times
for j in range(N_CF):
    # initialising starting path
    start_p = init
    # applying metropolis to path N_cor times
    for i in range(U):
        new_p, counts = Metropolis(start_p, pot)
        start_p = new_p
        t_counts += counts

    # adding final path to all_ps
    all_ps.append(start_p)
    plt.plot(start_p, t, alpha = 0.1)
#plt.show()

'gg'

