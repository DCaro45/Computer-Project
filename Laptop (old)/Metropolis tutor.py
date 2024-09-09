# %% Importing required modules
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing


# %% Defining various functions
def V(m, x):
    return 0.5 * m * x ** 2


def delta_action(xp, x1, x0, xu, V):
    s1 = 0.5 * m * (1 / a) * ((x1 - xu) ** 2 + (x0 - xu) ** 2) + a * V(m, xu)
    s2 = 0.5 * m * (1 / a) * ((x1 - xp) ** 2 + (x0 - xp) ** 2) + a * V(m, xp)
    return s2 - s1


def sweep(path, N, V):
    #    this is a single sweep, path is N long
    for i in range(N):

        x_perturbed = path[i] + (np.random.rand() - 0.5)
        print(np.random.rand)
        s_delta = delta_action(x_perturbed, path[(i + 1) % N], path[(i - 1) % N], path[i],
                               V)  # Calculates the change in action

        # Determines whether to keep the change
        if (s_delta < 0):
            path[i] = x_perturbed
        elif (np.random.rand() < np.exp(-s_delta)):
            path[i] = x_perturbed

    return path


def fpath(path_array):
    path = np.ones(N) * 5  # np.zeros(N)  #this is the initial path/array/condition
    #    path = (np.random.rand(N) - 0.5) * 2
    #    print('initial path:', path)

    Ncor = 10  # round(1 / (a**2)) + 1
    #    print('Ncor=', Ncor)

    #   this is the thermalisation stage (building distribution from which we can sample)
    #    print('you are thermalising')
    for i in range(Ncor):
        path = sweep(path, N, V)
    #        print('thermalising path', path)

    #        now comes rhe sampling, we are buiding an aray of shape N x runs
    #    print('you stopped thermalising')
    #    print('you stopped thermalising', path)
    for i in range(0, runs):
        for j in range(Ncor):
            path = sweep(path, N, V)
        #            print('path =', path)
        path_array.append(list(sweep(path, N, V)))
        #print('patharray =', path_array[-1])


# %% Assigning variables
start = time.time()
print("Calculating Path Integral")
m = 2
N = 7
T = 4
a = T / N
runs = 2000  # 200
threads = 5

# %% Thread initialisation
if __name__ == '__main__':
    path_array = multiprocessing.Manager().list()
    jobs = []
    for i in range(0, threads):
        p = multiprocessing.Process(target=fpath, args=(path_array,))
        jobs.append(p)
        p.start()

    while True:
        done = True
        for job in jobs:
            job.join(timeout=0)
            if job.is_alive():
                done = False
                break
        if done == True:
            break
        time.sleep(1)

    # %% Generating graph
    mid = time.time()
    #print("Time Taken to Calculate", runs * 10, "Runs:")
    #print(mid - start)

    flat_array = np.empty([runs * threads, N], dtype=float)
    i = 0
    for array in path_array:
        flat_array[i] = array
        i += 1
    flat_array = flat_array.flatten()

    def pdf(x):
        """prob density function"""

        prob = (np.exp(- x ** 2 / 2) / np.pi ** (1 / 4)) ** 2
        return prob

    #print(flat_array)
    xs = np.linspace(min(flat_array), max(flat_array), len(flat_array))
    pdf_A = pdf(xs)

    #print('flattened array=', flat_array)
    plt.hist(flat_array, bins=100, density=True, histtype="step")
    plt.plot(xs, pdf_A, label='PDF', color='tab:orange')
    plt.xlabel("x")
    plt.ylabel(r"$|\psi_0|^2$")
    #plt.savefig("Harmonic_oscillator_ground_state_1000000_runs.png", dpi=400)
    plt.show()

    end = time.time()
    #print("Time Taken to Generate Graph:")
    #print(end - mid)
    #print("Total Time Taken:")
    #print(end - start)