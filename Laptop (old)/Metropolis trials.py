"""operating at each point on path, keep each path iteration, start at the origin"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1  # setting mass to be 1

ti = 0   # start time
tf = 4   # finish time

div_t = 2  # division of time points (i.e whole numbs, half, third etc))

epsilon = 0.5  # change in delta_xs size from spatial lattice spacing
bins = 100  # number of bins for histogram

'''determinants/shorthands'''
n_t = div_t * (tf - ti) + 1  # number of spatial points
n_t = 7
a = (tf - ti) / (n_t - 1)  # size of time step
t_points = np.linspace(ti, tf, n_t)  # temporal lattice points

N_cor = 10            # number of paths to be skipped path set (due to correlation)
Sweeps1 = 10 * N_cor  # number of sweeps through path set
Sweeps2 = N_cor       # not necessary right now
N_CF = 10 ** 5           # number of updates

m = mass  # shorthand for mass
nt = n_t  # shorthand for no.t points
t = t_points  # shorthand for temporal lattice points
S1 = int(Sweeps1)  # shorthand for sweeps 1 (and integer data type)
S2 = int(Sweeps2)  # shorthand for sweeps 2 (and integer data type)


def pot(x):
    """simple harmonic oscillator potential"""
    V = 1/2 * x ** 2
    return V

def actn(x, p1, p2, potential):
    """calculating energies"""
    KE = 0.5 * m * (1/a) ** 2 * ((p2 - x) ** 2 + (p1 - x) ** 2)
    PE = potential(x)
    E_tot = KE + PE
    Action = a * E_tot

    '''
    k1 = 0.5 * m * (1 / a ** 2) * ((p2 - x) + (x - p1)) ** 2
    v1 = potential(x)
    s1 = a * (k1 + v1)
    '''

    return Action
    #return s1

def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    eval_p = path
    e = 1
    count = 0

    S1 = []
    S2 = []
    dS = []

    for j,x in enumerate(eval_p):
        dx = rdm.uniform(-e, e)
        xP = x + 0.1

        xn = eval_p[(j+1)%nt]
        xp = eval_p[(j-1)%nt]


        s1 = actn(x, xp, xn, potential)
        s2 = actn(xP, xp, xn, potential)

        ds = s2 - s1

        S1.append(s1)
        S2.append(s2)
        dS.append(ds)

        r = rdm.random()
        W = np.exp(-ds)
        if ds < 0:
            eval_p[j] = xP
            count += 1
        elif W > r:
            eval_p[j] = xP
            count += 1
        else:
            eval_p[j] = x

    return list(eval_p), S1, S2, dS

def V(m, x):
    return 0.5 * m * x ** 2

def delta_action(xp, x1, x0, xu, V):
    s1 = 0.5 * m * (1 / a) * ((x1 - xu) ** 2 + (x0 - xu) ** 2) + a * V(m, xu)
    s2 = 0.5 * m * (1 / a) * ((x1 - xp) ** 2 + (x0 - xp) ** 2) + a * V(m, xp)
    return s2 - s1, s1, s2

def sweep(path3, N, V):
    #    this is a single sweep, path is N long
    S1 = []
    S2 = []
    dS = []

    for i in range(N):

        x_perturbed = path3[i] + 0.1

        s_delta, s1, s2 = delta_action(x_perturbed, path3[(i + 1) % N], path3[(i - 1) % N], path3[i],
                               V)  # Calculates the change in action
        S1.append(s1)
        S2.append(s2)
        dS.append(s_delta)

        # Determines whether to keep the change
        if (s_delta < 0):
            path3[i] = x_perturbed
        elif (np.random.rand() < np.exp(-s_delta)):
            path3[i] = x_perturbed

    return path3, S1, S2, dS


time_step = 10
def energy(mass, x, j):
    N = len(x)
    eps = a
    jp = (j+1)%N
    jm = (j-1)%N

    pe = V(mass, x[j])
    ke = x[j]*(x[j]-x[jp]-x[jm]) / (eps**2)
    S = a * (pe + ke)

    return S

def therm_sweep(starting_points, acceptance_ratio):
    S1 = []
    S2 = []
    dS = []
    for i in range(len(starting_points)):
        initial_points = starting_points.copy()
        initial_energy = energy(1, initial_points, i)

        perturbed_points = starting_points.copy()
        perturbed_points[i] += 0.1
        perturbed_energy = energy(1, perturbed_points, i)

        energy_diff = perturbed_energy - initial_energy

        S1.append(initial_energy)
        S2.append(perturbed_energy)
        dS.append(energy_diff)

        eps = a
        if energy_diff < 0 or np.random.uniform(0,1) < np.exp(-eps * energy_diff):
            acceptance_ratio.append(1)
            starting_points[i] = perturbed_points[i]
        else:
            acceptance_ratio.append(0)

    return starting_points, S1, S2, dS


print('nt = ' + str(nt) + ',', 't = ' + str(t) + ',', 'S1 = ' + str(S1) + ',', 'N_cor = ' + str(N_cor))


accept = []

p_1 = [0 for x in range(nt)]
p1, S_1, S_2, dS_1 = Metropolis(p_1, pot)
p_1 = [0 for x in range(nt)]
p2, S_3, S_4, dS_2 = sweep(p_1, nt, V)
p_1 = [0 for x in range(nt)]
p3, S_5, S_6, dS_3 = therm_sweep(p_1, accept)



print(p1, p3)
print(S_1, S_3, S_5)
print(S_2, S_4, S_6)
print(dS_1, dS_2, dS_3)

for i in range(len(dS_1)):
    if dS_1[i] == dS_2[i] == dS_3[i]:
        print(True)
    else:
        print(False)


def path_check(path_old, path_new):
    return

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