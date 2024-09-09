"""operating at each point in the path, keeping final set of paths"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt

'''values'''
mass = 1  # setting mass to be 1

x0 = -2  # lower bound
x1 = 2  # upper bound
ti = 0  # start time
tf = 5  # finish time

div_t = 2  # division of time points (i.e whole numbs, half, third etc))
div_x = 2  # division of space points
N = 10 ** 3  # number of paths seeding metropolis

_shrink_ = 0.35  # change in delta_xs size from spatial lattice spacing
bins = 100  # number of bins for histogram

'''determinants/shorthands'''
n_t = div_t * (tf - ti) + 1  # number of spatial points
n_x = div_x * (x1 - x0) + 1  # number of spatial points
a = (tf - ti) / (n_t - 1)  # size of time step
b = (x1 - x0) / (n_x - 1)  # size of spacial step
t_points = np.linspace(ti, tf, n_t)  # temporal lattice points
x_points = np.linspace(x0, x1, n_x)  # spacial lattice points

N = int(N)  # so python identifies N as an integer
N_cor = 1 / (a ** 2)  # number of paths to be skipped path set (due to correlation)
Sweeps1 = 10 ** 2 * N_cor  # number of sweeps through path set
Sweeps2 = N_cor  # not necessary right now

m = mass  # shorthand for mass
nt = n_t  # shorthand for no.t points
nx = n_x  # shorthand for no.x points
x = x_points  # shorthand for spatial lattice points
t = t_points  # shorthand for temporal lattice points
S1 = int(Sweeps1)  # shorthand for sweeps 1 (and integer data type)
S2 = int(Sweeps2)  # shorthand for sweeps 2 (and integer data type)


def pot(x):
    """simple harmonic oscillator potential"""

    V = 1 / 2 * (x) ** 2
    return V


def path_gen2(xs, N):
    """path generator"""
    # nt is the length of the path
    N = int(N)
    div = N // nt
    rem = N % nt
    paths = np.zeros([N, nt])
    indx = 0
    if rem == 0:
        for x0 in xs:
            for m in range(div):
                for i in range(nt):
                    paths[indx][i] = x0
                indx += 1
    else:
        repeat = []
        for i in range(nt):
            repeat.append(0)
        while repeat.count(1) < rem:
            ix = rdm.randrange(0, nt)
            if repeat[ix] == 1:
                pass
            else:
                repeat[ix] = 1
        for i, x0 in enumerate(xs):
            for m in range(int(div + repeat[i])):
                for i in range(nt):
                    paths[indx][i] = x0
                indx += 1
    return paths


def actn(path, potential):
    """calculating energies"""
    # dont calculate change in energies for points that haven't changed
    E_path = 0
    for i in range(0, nt - 1):
        if path[i + 1] == path[i]:
            KE = 0
            PE = potential(path[i])
        else:
            KE = m / 2 * ((path[i + 1] - path[i]) / a) ** 2
            PE = potential((path[i + 1] + path[i]) / 2)
        E_tot = KE + PE
        E_path += E_tot
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
        normalised = array / total
        return normalised
    else:
        return 0


def Metropolis(path, potential):
    """creating the metropolis algorithm"""
    epsilon = b / _shrink_
    e = epsilon
    count = 0
    eval_p = path
    # append each path onto a large list

    for j in range(nt):
        old_p = eval_p
        S_old = actn(old_p, potential)

        new_p = np.zeros([nt])
        dx = rdm.uniform(-e, e)
        for i, x in enumerate(old_p):
            if i == j:
                new_p[i] = x + dx
            else:
                new_p[i] = x
        S_new = actn(new_p, potential)
        dS = S_new - S_old

        r = rdm.random()
        W = wght(dS)
        if dS < 0:
            eval_p = new_p
            count += 1
        elif W > r:
            eval_p = new_p
            count += 1
    return eval_p, count


print('nx = ' + str(nx) + ',', 'nt = ' + str(nt) + ',', 'x = ' + str(x) + ',', 't = ' + str(t) + ',',
      'N = ' + str(N) + ',',
      'S1 = ' + str(S1) + ',', 'N_cor = ' + str(N_cor))



p_1 = path_gen2(x, 5)
print(p_1[0])
new_p, count = Metropolis(p_1[0], pot)
print(new_p, count, (count/nt))
old_ps = path_gen2(x, N)


"""generating paths and applying metropolis"""
new_ps1 = old_ps
t_counts = 0
for i in range(S1):
    for j in range(N):
        new_ps1[j], counts = Metropolis(old_ps[j], pot)
        t_counts += counts
print(new_ps1, 'points calc')
print('prop of changing point = ' + str(t_counts/(nt*N*S1)))


"""points from new_ps skipping every N_cor'th one"""
pos_1 = np.zeros([int(N/N_cor) * nt])
print(N_cor, len(new_ps1) * nt, len(pos_1))
m = 0
for i in range(int(N/N_cor)):
    for j in range(nt):
        pos_1[m] = new_ps1[int(N_cor*i)][j]
        m += 1
print(pos_1, 'positions calc')
#print(pos, len(new_ps),len(pos))


xs = np.linspace(min(pos_1), max(pos_1), len(pos_1))
pdf_A = pdf(xs)
Norm = norm(pdf_A)

counts, bins = np.histogram(pos_1, bins = bins)
y2 = (max(counts)/max(Norm))*norm(pdf_A)

plt.hist(pos_1, bins = bins, label = 'M2')
plt.plot(xs , y2, label = 'PDF', color = 'tab:orange' )
plt.legend()
plt.show()