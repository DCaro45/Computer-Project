"""calculating the energy difference between the ground and first excited state of a simple harmonic oscillator"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

dir, file = os.path.split(__file__)
save = False


'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 20     # finish time
step = 0.5   # division of time points (i.e whole numbs, half, third etc))

epsilon = 1.8       # change in delta_xs size from spatial lattice spacing
N_cor = 25          # number of paths to be skipped path set (due to correlation)
N_CF = 10 ** 5      # number of updates
Therm = 5 * N_cor   # number of sweeps through path set

w = 1     # harmonic constant
L = 3     # well length


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


def pot1(x):
    V = 1/2 * x ** 2
    return V


def pot2(x):
    V = 1/2 * w**2 * x ** 2
    return V


def pot3(x):
    l = L/2
    if -l < x < l:
        V = 0
    else:
        V = 100000000
    return V


def po4(x):
    """a potential with the same form as the higgs potential"""
    u = 2
    l = 1
    V = - 0.5 * u ** 2 * x ** 2 + 0.25 * l ** 2 * x ** 4
    return V


def pot5(x):
    a = 1
    b = -1/12
    c = 1/1500
    d = 1/10000
    p = 1/8
    V = a * x + b * x ** 3 + c * x ** 5 + d * np.exp(x ** 2) ** p
    return V


def pot6(x):
    """ a polynomial potential with a minimum and a stationary inflection point"""
    V = - 3 * x ** 2 - 1/4 * x ** 3 + 1/2000 * x ** 6
    return V

def actn(x, j, potential):
    "calculating energies"
    # setting index so that it loops around
    jp = (j-1) % nt
    jn = (j+1) % nt

    # calculating energies ... strange???
    KE = m * x[j] * (x[j] - x[jp] - x[jn]) / (a ** 2)
    PE = potential(x[j])
    E_tot = KE + PE
    Action = a * E_tot

    return Action

def Metropolis(path, potential):
    "creating the metropolis algorithm"
    # keeping count of number of changes
    count = 0

    for j, x in enumerate(path):
        # creating a perturbed path from initial path
        dx = rdm.uniform(-e, e)
        xP = x + dx
        eval_p = path.copy()
        eval_p[j] = xP

        # calculating actions
        S1 = actn(path, j, potential)
        S2 = actn(eval_p, j, potential)
        ds = S2 - S1

        # applying metropolis logic
        r = rdm.random()
        W = np.exp(-ds)
        if ds < 0 or W > r:
            path = eval_p
            count += 1

    return path, count

def compute_G1(x,k):
    g = 0
    for j in range(nt):
        jn = (j+k) % nt
        g += x[j] * x[jn]
    return g/nt

def compute_G2(x,k):
    g = 0
    for j in range(nt):
        jn = (j+k) % nt
        g += x[j] ** 3 * x[jn] ** 3
    return g/nt

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
pot = pot2


p_1 = [0 for t in range(nt)]
p1, count = Metropolis(p_1, pot)


"""Thermalising lattice"""
init = p_1
for i in range(T):
    x, count = Metropolis(init, pot)
    init = x
therm = init

"""generating array of G values"""
G = np.zeros([N, nt])
count = 0
x = therm
for alpha in range(N):
    for j in range(U):
        new_x, c = Metropolis(x, pot)
        x = new_x
        count += c
    for k in range(nt):
        G[alpha][k] = compute_G(x,k)
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
Avg_G = avg(G)

'Binning G values'
binned_G = bin(G, 20)
B = binned_G
Avg_B = avg(B)


print('done dE')

"""Calculating errors"""
'Bootstrap'
M = int(10e+2)
dE_bootstrap = np.zeros([M, nt - 1])
for i in range(M):
    G_bootstrap = bootstrap(G)
    Avg_G = avg(G_bootstrap)
    dE_bootstrap[i] = delta_E(Avg_G)
    if i % p == 0:
        print(i/M * 100)
dE_avg = avg(dE_bootstrap)
dE_sd = sdev(dE_bootstrap)

"""
'Binned'
nums = [1.5**i for i in range(1, 18)]
print(nums)
sd_bin = np.zeros([len(nums), nt - 1])
for i, m in enumerate(nums):
    b = bin(G, m)
    '''
    avg_b = avg(b)
    deltaE = delta_E2(avg_b)
    sd = sdev(deltaE)
    sd_bin[i] = sdev(deltaE)
    '''
    'Bootstrap'
    M = int(10e+1)
    dE_bootstrap = np.zeros([M, nt - 1])
    for j in range(M):
        G_bootstrap = bootstrap(b)
        Avg_G = avg(G_bootstrap)
        dE_bootstrap[j] = delta_E(Avg_G)
    sd = sdev(dE_bootstrap)
    sd_bin[i] = sd
print('done bootstrap')
"""

if pot == pot1:
    name = 'Harmonic'
elif pot == pot2:
    name = 'Harmonic-' + str(w)
elif pot == pot3:
    name = 'box' + str(L)

"""Plotting"""

plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

ts = t[:-1]
plt.figure(figsize=(8, 4))
if pot == pot1 or pot == pot2:
    dE_analytic = [w for t in range(nt - 1)]
    plt.plot(ts, dE_analytic, linestyle='--', color='black', label='Analytic solution')
plt.errorbar(ts, dE_avg, yerr=dE_sd, color='green', fmt='o', capsize=4, elinewidth=1, label='Delta_E')
print(dE_avg[0])

plt.xlabel('t')
plt.ylabel('$\Delta$E(t)')
plt.legend()
plt.title('1D-Delta_E_' + name, y=1.075)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False, direction='in')
plt.tick_params(bottom=True, top=False, left=True, right=False)
if save == True:
    plt.savefig(dir + '\\Images\\DeltaE_1D-' + name + '.png')
plt.show()



