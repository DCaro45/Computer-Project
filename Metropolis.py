"""Using the Metropolis algorithm to calculate the probability density function of a harmonic oscillator potential"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

from scipy.stats import norm

dir, file = os.path.split(__file__)
save = False


'''values'''
mass = 1   # setting mass to be 1

t_i = 0       # start time
t_f = 4       # finish time
step = 0.5    # division of time points (i.e whole numbs, half, third etc))

epsilon = 10   # change in delta_xs size from spatial lattice spacing
bins = 500     # number of bins for histogram

N_cor = 20         # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor  # number of sweeps through path set
N_CF = 10 ** 6     # number of updates

'''determinants/shorthands'''
t_points = np.arange(t_i, t_f + step, step)  # number of temporal points
n_tp = len(t_points)          # number of temporal points
n_tl = n_tp - 1  # temporal lattice points


m = mass           # shorthand for mass
nt = n_tp           # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
a = step
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print('nt = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' +
      'N_cor/Update = ' + str(N_cor) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N))

def Res(y, y_mdl):
    R = y - y_mdl
    return R

def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V


def pdf(x):
    """prob density function"""
    prob = np.exp(- x ** 2) / (np.pi ** (1/2))
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


"""Initialising paths and trialing metropolis"""

p_1 = [0 for x in range(nt)]
p1, count = Metropolis(p_1, pot)
print(p1, count/nt)


"""Thermalising lattice"""

init = p_1
for i in range(T):
    new_p, counts = Metropolis(init, pot)
    init = new_p

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


xs = np.linspace(-3, 3, len(pos))
PDF = pdf(xs)
V = pot(xs)

counts, bins = np.histogram(pos, bins=bins)
diff = np.diff(bins)[0]
i = 0
for j, pos in enumerate(bins):
    if -diff <= pos <= diff:
        i = j
        break
Norm = max(PDF) * counts/counts[i]

x = bins[:-1] + np.diff(bins)/2
modl = pdf(x)
y = Norm
res = Res(y, modl)
mean = np.mean(res)
std = np.std(res)
x_hist = np.arange(-5, 5, 0.1)
gauss = norm.pdf(x_hist, mean, std)
lim1, lim2 = -3, 3


plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

fig = plt.figure(figsize=[12,6])

ax1 = fig.add_axes((0.07, 0.295, 0.87, 0.65))
ax1.tick_params(labelbottom=False, labeltop=False, labelleft=True, labelright=False, direction='in')
ax1.tick_params(bottom=True, top=False, left=True, right=False)

ax2 = ax1.twinx()

ax3 = plt.gca()
ax3 = fig.add_axes((0.07, 0.1, 0.87, 0.17))
ax3.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
ax3.tick_params(bottom=True, top=True, left=True, right=False)


ax1.stairs(Norm, bins, fill=True, label='Monte Carlo integral')
ax1.plot(xs, PDF, color='black', label='Analytic solution')
ax2.plot(xs, V, color='red', label='Potential')
ax3.scatter(x, res)

ax3.axhline(y=0, linestyle='dashed', color='black', linewidth='1')
ax3.axhline(y=5, color='darkgrey', linewidth='1')
ax3.axhline(y=-5, color='darkgrey', linewidth='1')
ax3.axhspan(-5, 5, color='lightgrey', alpha=0.3)


ax1.set_xlim([lim1, lim2])
ax1.set_ylabel('|' + chr(968) + '|' + chr(178), color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.legend()

ax2.set_ylim([0, max(V)])
ax2.set_ylabel('Potential', color='black', rotation=270)
ax2.yaxis.set_label_coords(1.05,0.5)
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper left')

ax3.set_xlim([lim1, lim2])
ax3.set_ylim([-0.015, 0.015])
ax3.set_xlabel("x")
ax3.set_ylabel('\n Residuals')


fig.tight_layout()
if save == True:
    fig.savefig(dir + '\\Images\\Hist-Harmonic.png')
plt.show()

