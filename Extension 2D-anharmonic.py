"""using metropolis to find probability density function for 2D potentials"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os

from Demos.win32cred_demo import save
from matplotlib import cm

dir, file = os.path.split(__file__)
plot = False
save = False

'''values'''
mass = 1   # setting mass to be 1

t_i = 0      # start time
t_f = 10     # finish time
step = 1     # division of time points (i.e. whole numbs, half, third etc.))

epsilon = 2    # change in delta_xs size from spatial lattice spacing
bins = 50      # number of bins for histogram

N_cor = 20         # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor  # number of sweeps through path set
N_CF = 10 ** 6     # number of updates

u = 2            # potential parameter 1
l = 1            # potential parameter 2

'''determinants/shorthands'''
t_points = np.arange(t_i, t_f + step, step)  # number of temporal points
n_tp = len(t_points)          # number of temporal points
n_tl = n_tp - 1               # number of temporal links

m = mass           # shorthand for mass
nt = n_tp          # shorthand for no.t points
t = t_points       # shorthand for temporal lattice points
e = epsilon        # shorthand for epsilon
a = step           # shorthand for size of time step
U = int(N_cor)     # shorthand for sweeps 2 (and integer data type)
T = int(Therm)     # shorthand for sweeps 1 (and integer data type)
N = int(N_CF)      # shorthand for number of updates

print(
    'n = ' + str(nt) + ', ' + 'a = ' + str(a) + ', ' + 't = ' + str(t) + ', ' + 'epsilon = ' + str(e) + ', ' +
    'N_cor/Update = ' + str(U) + ', ' + 'S1 = ' + str(T) + ', ' + 'N_CF = ' + str(N)
      )


def pdf(x, y):
    """prob density function"""
    r = np.sqrt(x ** 2 + y ** 2)
    prob = np.exp(- r ** 2) / np.pi
    return prob

def circ(pos, r):
    x, y = pos
    rho = (x ** 2 + y ** 2) ** (1 / 2)
    if rho <= r:
        return 1
    else:
        return 0

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
    V = np.cos(2 * x) + np.sin(2 * y) + 1/5000 * np.exp(x**2) + 1/5000 * np.exp(y**2)
    return V

def pot5(x, y):
    length = 5
    l = length/2
    if -l < x < l and -l < y < l:
        V = 0
    else:
        V = 100000000
    return V


pot = pot3

x0 = -3
x1 = 3
L = 250
X = np.linspace(x0, x1, L)
Y = np.linspace(x0, x1, L)
X, Y = np.meshgrid(X, Y)
if pot != pot5:
    Z = pot(X, Y)
if pot == pot5:
    Z = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            Z[i, j] = pot(X[i, j], Y[i, j])
if plot == True:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.tick_params(axis='z', labelcolor='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Potential')
    plt.show()

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

"""Initialising paths and trialing metropolis"""

px_1 = np.zeros([nt])
py_1 = np.zeros([nt])
px1, py1, count = Metropolis(px_1, py_1, pot)
print(px1, py1, count/nt, sep='\n')

"""Thermalising lattice"""

init_x = px_1
init_y = py_1
'''
for i in range(T):
    new_px, new_py, counts = Metropolis(init_x, init_y, pot)
    init_x = new_px
    init_y = new_py
'''


"""Generating paths and applying metropolis"""

all_px = np.zeros([N, nt])
all_py = np.zeros([N, nt])
all_px[0] = init_x
all_py[0] = init_y

t_counts = 0
for j in range(N - 1):
    start_px = all_px[j]
    start_py = all_py[j]
    for i in range(U):
        new_px, new_py, counts = Metropolis(start_px, start_py, pot)
        start_px = new_px
        start_py = new_py
        t_counts += counts
    all_px[j + 1] = start_px
    all_py[j + 1] = start_py
    p = N/100
    if j % p == 0:
        print(j/N * 100)
print('prop of changing point = ' + str(t_counts/(nt*U*N)))


"""All points from new_ps"""

xpos = np.zeros([N * nt])
ypos = np.zeros([N * nt])

k = 0
for i in range(N):
    for j in range(nt):
        xpos[k] = all_px[i][j]
        ypos[k] = all_py[i][j]
        k += 1
print('done')



"""Graphs"""

"Generating potential"
x0 = min(xpos)
x1 = max(xpos)
y0 = min(ypos)
y1 = max(ypos)

N = 500
X = np.linspace(x0, x1, N)
Y = np.linspace(y0, y1, N)
X, Y = np.meshgrid(X, Y)

if pot != pot5:
    Z = pot(X, Y)
if pot == pot1:
    name = 'Higgs'
elif pot == pot2:
    name = 'poly'
elif pot == pot3:
    name = 'Anharmonic_1'
elif pot == pot4:
    name = 'Anharmonic_2'
elif pot == pot5:
    name = 'Box'
    Z = np.zeros([L, L])
    for i in range(L):
        for j in range(L):
            Z[i, j] = pot(X[i, j], Y[i, j])

"Generate 3D histogram"

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins, density=True)
Z = Z * (np.max(hist)/np.max(Z))


"Plotting 3D Histogram"

plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
x = x.flatten()/2
y = y.flatten()/2
z = np.zeros_like(x)

dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
x = x - dx/2
y = y - dy/2
dz = hist.flatten()

cmap = cm.viridis
max_height = np.max(dz)
min_height = np.min(dz)
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(x, y, z, dx, dy, dz, color=rgba, zsort='average')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.3, linewidth=0, antialiased=False)

plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('|' + chr(968) + '|' + chr(178))
#ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')
if save == True:
    fig.savefig(dir + '\\Images\\3Dhist-' + name + '-' + str(t_f) + 's.png')
plt.show()
ind = np.argpartition(hist, -4)[-4:]
print(ind)


"Contour Hist"

x, y = np.meshgrid(xedges, yedges)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
ax.pcolormesh(x, y, hist.T)

min = 0.0001
if pot == pot3:
    min = 0.000005
if pot == pot4:
    min = 0.00001

if pot != pot1:
    i = np.where(Z <= np.min(Z)+min)
    x_i = i[0]
    y_i = i[1]
    x = X[x_i, x_i]
    y = Y[y_i, y_i]
    ax.scatter(x, y, color='black', s=10)
    plt.show()

if pot == pot1:
    R = u/l
    N = 250
    xs = np.linspace(x0, x1, N)
    ys = np.linspace(x0, x1, N)
    dat1 = np.zeros((N, N))
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            pos = [x, y]
            dat1[iy, ix] = circ(pos, R)
    plt.contour(xs, ys, dat1, 1, cmap='binary')
    plt.show()
if save == True:
    fig.savefig(dir + '\\Images\\contour_hist-' + name + '-' + str(t_f) + 's.png')
plt.show()
