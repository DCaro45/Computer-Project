"""using metropolis to find probability density function for 2D potentials"""

import numpy as np
import random as rdm
import matplotlib.pyplot as plt
import os
from matplotlib import cm

plt.rcParams["font.family"]="serif"
plt.rcParams["mathtext.fontset"]="dejavuserif"

dir, file = os.path.split(__file__)

'''values'''
mass = 1   # setting mass to be 1

t_i = 0     # start time
t_f = 10     # finish time
step = 1  # division of time points (i.e. whole numbs, half, third etc.))

epsilon = 1    # change in delta_xs size from spatial lattice spacing
bins = 75      # number of bins for histogram

N_cor = 20         # number of paths to be skipped path set (due to correlation)
Therm = 5 * N_cor  # number of sweeps through path set
N_CF = 10 ** 5     # number of updates

save = True

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


def Res(z, z_mdl):
    R = z - z_mdl
    return R


def pdf(x, y):
    """prob density function"""
    r = np.sqrt(x ** 2 + y ** 2)
    prob = np.exp(- r ** 2) / np.pi
    return prob


def pot(x, y):
    """simple harmonic oscillator potential"""
    r = np.sqrt(x ** 2 + y ** 2)
    V = 1/2 * r ** 2
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

X = np.linspace(x0, x1, 100)
Y = np.linspace(y0, y1, 100)
X, Y = np.meshgrid(X, Y)
Z = pdf(X, Y)
name = 'Harmonic'

"Generate 3D histogram"

hist, xedges, yedges = np.histogram2d(xpos, ypos, bins=bins)
diff_x = np.diff(xedges)[0]
diff_y = np.diff(yedges)[0]
i = 0
j = 0
for k, pos in enumerate(xedges):
    if -diff_x <= pos <= diff_x:
        i = k
        break
for k, pos in enumerate(yedges):
    if -diff_y <= pos <= diff_y:
        j = k
        break
Norm = np.max(Z)/hist[i,j] * hist

"Generating residuals"

xs = xedges[:-1] + np.diff(xedges)/2
ys = yedges[:-1] + np.diff(yedges)/2

xs, ys = np.meshgrid(xs, ys)
v = pdf(xs, ys)
z = Norm
res = Res(z, v)
res_x = np.zeros([len(xs)])
res_y = np.zeros([len(ys)])
for i in range(len(xs)):
    res_x[i] = np.mean(res[i, :])
    res_y[i] = np.mean(res[:, i])

"Plotting 3D Histogram"
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x, y = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
x = x.flatten()/2
y = y.flatten()/2
z = np.zeros_like(x)


dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]
x = x - dx/2
y = y - dy/2
dz = Norm.flatten()


cmap = cm.viridis
max_height = np.max(dz)
min_height = np.min(dz)
rgba = [cmap((k-min_height)/max_height) for k in dz]

ax.bar3d(x, y, z, dx, dy, dz, color=rgba, zsort='average')
ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.2, linewidth=0, antialiased=False)

z_pos = - 0.05
ax.scatter(xs[0,:] + np.diff(xedges), min(yedges) - np.diff(yedges)[0]/2, (5*res_x) + z_pos, marker='o', color='black')
ax.scatter(min(xedges) - np.diff(xedges)[0]/2, ys[:,0] + np.diff(yedges), (5*res_y) + z_pos, marker='o', color='black')
x_dash = np.arange(xs[0,0] + np.diff(xedges)[0], xs[-1,-1] + np.diff(xedges)[0], 0.25)
y_dash = np.arange(ys[0,0] + np.diff(yedges)[0], ys[-1,-1] + np.diff(yedges)[0], 0.25)
ax.scatter(x_dash, min(yedges) - np.diff(yedges)[0]/2, z_pos, marker='_', color='grey')
ax.scatter(min(xedges) - np.diff(xedges)[0]/2, y_dash, z_pos, marker='_', color='grey')


plt.xlabel("x")
plt.ylabel("y")
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlabel('|' + chr(968) + '|' + chr(178))
#ax.zaxis.set_major_formatter('{x:.02f}')
ax.tick_params(axis='z', labelcolor='red')
if save == True:
    fig.savefig(dir + '\\Images\\3Dhist_' + name + '-' + str(t_f) + 's_new.png')
plt.show()


"Contour Hist"
x, y = np.meshgrid(xedges, yedges)

# Histogram contour
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
ax.pcolormesh(x, y, Norm.T)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
if save == True:
    fig.savefig(dir + '\\Images\\contour-hist_' + name + '-' + str(t_f) + 's_new.png')
plt.show()

# Residual contour
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()
ax.pcolormesh(xs, ys, res)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
if save == True:
    fig.savefig(dir + '\\Images\\contour-res_' + name + '-' + str(t_f) + 's_new.png')
plt.show()

