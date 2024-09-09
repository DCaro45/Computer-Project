"""calculating the energy difference between the ground and first excited state of a simple harmonic oscillator"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

dir, file = os.path.split(__file__)

'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 100     # finish time
step = 2.5   # division of time points (i.e whole numbs, half, third etc))

epsilon = 0.8         # change in delta_xs size from spatial lattice spacing
N_cor = 25          # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5      # number of updates
Therm = 5 * N_cor   # number of sweeps through path set

u = 2
l = 1

w = 1   # harmonic constant
L = 6   # well length

save = True

'''determinants/shorthands'''
t_points = np.arange(t_i, t_f + step, step)  # number of temporal points
n_tp = len(t_points)          # number of temporal points
n_tl = n_tp - 1

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
a = step           # shorthand for size of time step
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print('# temporal points = ' + str(nt) + ', ' + 'size of time step = ' + str(a) + ', ' + 'temporal points = ' + str(t) +
      ', ' + 'epsilon = ' + str(e) + ', ' '# updates (N_cor) = ' + str(N_cor) + ', ' + 'Therm sweeps (T) = ' + str(T) +
      ', ' + '# paths (N_CF) = ' + str(N))

def pot00(x, y):
    """simple harmonic oscillator potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * r ** 2
    return V


def pot0(x, y):
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * w * r ** 2
    return V

def pot1(x, y):
    """a simple potential analogue for the Higgs potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = - 0.5 * u ** 2 * r ** 2 + 0.25 * l ** 2 * r ** 4
    return V


def pot2(x, y):
    """ an anharmonic potential with a variety of minima"""
    V = - 10 * x**2 - 8 * y**2 + 6 * x**4 + 3 * y**4 + 1/10 * x**6 + 1/10 * y**6
    return V


def pot3(x,y):
    """ an anharmonic potential with a variety of minima using sin and cos functions"""
    V = np.cos(2 * x) + np.cos(2 * y) + 1/5000 * np.exp(x**2) + 1/5000 * np.exp(y**2)
    return V

def pot4(x,y):
    """ an anharmonic potential with a variety of minima using sin and cos functions"""
    V = np.cos(2 * x) + 3 * np.sin(2 * y) + 1/5000 * np.exp(x**2) + 1/5000 * np.exp(y**2) - 0.5*x - 0.5*y
    return V

def pot5(x, y):
    l = L/2
    if -l < x < l and -l < y < l:
        V = 0
    else:
        V = 100000000
    return V

def actn(x, y, j, potential):
    """calculating energies"""
    jp = (j-1) % nt
    jn = (j+1) % nt

    KE_x = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    KE_y = m * y[j] * (y[j] - y[jp] - y[jn]) / (a ** 2)
    KE = KE_x + KE_y

    PE = potential(x[j], y[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action


def Metropolis(path_x, path_y, potential):
    """creating the metropolis algorithm"""
    count = 0
    N = len(path_x)
    M = len(path_y)

    for i in range(nt):
        if N != nt or M != nt:
            print('error: path length not equal to n')
            break
        dx = rdm.uniform(-e, e)
        dy = rdm.uniform(-e, e)

        eval_px = path_x.copy()
        eval_py = path_y.copy()
        eval_px[i] = path_x[i] + dx
        eval_py[i] = path_y[i] + dy

        S1 = actn(path_x, path_y, i, potential)
        S2 = actn(eval_px, eval_py, i,  potential)
        dS = S2 - S1

        r = rdm.random()
        W = np.exp(-dS)
        if dS < 0 or W > r:
            path_x = eval_px
            path_y = eval_py
            count += 1

    return path_x, path_y, count

def compute_G1(x, y, k):
    g = 0
    for j in range(nt):
        jn = (j+k) % nt
        g += x[j] * x[jn] + y[j] * y[jn]
    G = g/(nt**2)
    return G

def compute_G2(x, y, k):
    g = 0
    for j in range(nt):
        jn = (j+k) % nt
        g += x[j] ** 3 * x[jn] ** 3 + y[j] ** 3 * y[jn] ** 3
    G = g/(nt**2)
    return G

def avg(p):
    Av = sum(p)/len(p)
    return Av

def sdev(p):
    P = np.array(p)
    sd = np.absolute(avg(P) ** 2 - avg(P ** 2)) ** (1/2)
    return sd

def delta_E(prop):
    G = np.array(prop)
    dE = np.log(np.absolute(G[:-1]/G[1:])) / a
    return dE

def bootstrap(G):
    G_bootstrap = []
    L = len(G)
    for i in range(L):
        alpha = rdm.randint(0, L-1)
        G_bootstrap.append(G[alpha])
    return G_bootstrap

def bin(G, number):
    G_binned = []
    binsize = int(N/number)
    for i in range(0, N, binsize):
        G_avg = 0
        for j in range(binsize):
            if i+j >= N:
                break
            G_avg += G[i+j]
        G_avg = G_avg/binsize
        G_binned.append(G_avg)
    return G_binned

compute_G = compute_G1
pot = pot3


"""Initialising paths and trialing metropolis"""

px_1 = np.zeros([nt])
py_1 = np.zeros([nt])
px1, py1, count = Metropolis(px_1, py_1, pot)


"""Thermalising lattice"""

init_x = px_1
init_y = py_1
'''
for i in range(T):
    new_px, new_py, counts = Metropolis(init_x, init_y, pot)
    init_x = new_px
    init_y = new_py
'''
therm_x = init_x
therm_y = init_y

"""generating array of G values"""
G = np.zeros([N, nt])
count = 0
x = therm_x
y = therm_y
for alpha in range(N):
    for j in range(U):
        new_x, new_y, c = Metropolis(x, y, pot)
        x = new_x
        y = new_y
        count += c
    for k in range(nt):
        G[alpha][k] = compute_G(x, y, k)
    p = N/100
    if alpha % p == 0:
        print(alpha/N * 100)
print('prop of changing point = ' + str(count/(nt*U*N)))
print('done G')

"""averaging G values"""
Av_G = np.zeros([nt])
for k in range(nt):
    avg_G = 0
    for alpha in range(N):
        avg_G += G[alpha][k]
    avg_G = avg_G/N
    Av_G[k] = avg_G


"""Calculating delta_E"""
dE = delta_E(Av_G)    # delta_E for average G (using function)


print('done dE')

"""Calculating errors"""
'Bootstrap'
M = int(10e+2)
dE_bootstrap = np.zeros([M, nt - 1])
for i in range(M):
    G_bootstrap = bootstrap(G)
    Avg_G = avg(G_bootstrap)
    dE_bootstrap[i] = delta_E(Avg_G)
    p = M/100
    if i % p == 0:
        print(i/M * 100)
dE_avg = avg(dE_bootstrap)
dE_sd = sdev(dE_bootstrap)
print('done boot')


"""Plotting"""

if compute_G == compute_G1:
    name = 'x'
    c = 'green'
if compute_G == compute_G2:
    name = 'x^3'
    c = 'blue'

if pot == pot0:
    name = 'Harmonic-' + str(w)
elif pot == pot1:
    name = 'Higgs'
elif pot == pot2:
    name = 'poly'
elif pot == pot3:
    name = 'Anharmonic_1'
elif pot == pot4:
    name = 'Anharmonic_2'
elif pot == pot5:
    name = 'Box'

ts = t[:-1]
plt.figure(figsize=(8, 4))
if pot == pot0 or pot == pot00:
    dE_analytic = [w for t in range(nt - 1)]
    plt.plot(ts, dE_analytic, linestyle='--', color='black', label='Analytic solution')
plt.errorbar(ts, dE_avg, yerr=dE_sd, color=c, fmt='o', capsize=4, elinewidth=1, label='Delta_E')
print(dE_avg[0])

plt.xlabel('t')
plt.ylabel('$\Delta$E(t)')
plt.legend()
plt.title('2D Delta_E-' + name, y=1.075)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
plt.tick_params(bottom=True, top=False, left=True, right=False)
if save == True:
    plt.savefig(dir + '\\Images\\DeltaE_2D-' + name + '.png')
plt.show()



