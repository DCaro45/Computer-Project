import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpt
import random as rdm
from matplotlib import rc

##values

x0 = -2       #lower bound
x1 = 2        #upper bound
ti = 0        #start time
tf = 5        #finish time

n_t = 6                         #Number of time points
divs_x = 2                      #Division of space points (i.e whole numbs, half, third etc)


##determinants

n_x = divs_x * (x1 - x0) + 1         #number of spatial points
a = (tf-ti)/(n_t-1)                  #size of time step
b = (x1-x0)/(n_x-1)                  #size of spacial step
t_points = np.linspace(ti, tf, n_t)  #time lattice points
x_points = np.linspace(x0, x1, n_x)  #spacial lattice points

nx = n_x                          #shorthand for no.x points
nt = n_t
x = x_points
t = t_points

print(x, t)
print(nt)

##path

def path_gen(xs, n, x0):
    path = np.zeros([n])
    path[0]=path[n-1] = x0
    for i in range(1,n-1):
        path[i]=rdm.choice(xs)
    return path

p_1 = path_gen(x, nt, x[2])
print(p_1)

###graph

samples = 100

#time points
ts = np.linspace(ti, tf, nt)

plt.figure(figsize = [10, 4])

for i in range(0,samples):
    p = path_gen(x, nt, x[4])
    plt.plot(t, p, alpha = 0.25)

plt.grid(axis = 'x')
yticks = np.arange(-2, 2.1, 1)
plt.yticks(yticks)
plt.xlabel('time')
plt.ylabel('position')
plt.show()
##repeat for many paths, use same starting point

Ns = np.logspace(start=1, stop= 6, base=10, num= 6)

print(Ns)



