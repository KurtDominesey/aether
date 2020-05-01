import sys
import copy

import numpy as np
import matplotlib.pyplot as plt

import h5py
import openmc.mgxs

def plot_spectrum(filename, savename, **kwargs):
    file = h5py.File('/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/' +
                     'mgxs-SHEM-361.h5', 'r')
    materials = ['void', 'water', 'uo2']
    table = np.genfromtxt(filename, names=True)
    xdata = openmc.mgxs.GROUP_STRUCTURES['SHEM-361'][:-1][::-1]
    plt.xscale('log')
    for name in table.dtype.names:
        kwargs_line = copy.deepcopy(kwargs)
        ydata = table[name]
        ydata /= np.linalg.norm(ydata, 2)
        if 'j' in name:
            j = int(name.split('j')[1])
            # if j < 1 or j > 2:
            if j != 2:
                continue
            material = materials[j]
            # if material == 2:
            #     kwargs['ls'] = '--'
        # if name == 'j0' or name == 'j3':  # void
        #     continue
        elif name == 'inf':
            pass
            # kwargs_line['ls'] = ':'
        # xdata = range(len(ydata))
        ydata *= file['uo2']['294K']['total']
        plt.step(xdata, ydata, where='mid', label=name, **kwargs_line)
    plt.yscale('log')
    plt.legend(loc='best')
    groups_coarse = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']
    for g_rev, bound in enumerate(groups_coarse):
        g = (len(groups_coarse)-1) - g_rev
        if (bound > 3.5 and bound < 400):
            plt.axvline(bound, ls='--', color='black', label=g)
    plt.xlim(3, 500)
    # twinx = plt.gca().twinx()
    # twinx.step(xdata, total, where='mid', color='C4', 
    #            alpha=0.5, label='uo2 xs_t')
    # plt.yscale('log')
    # plt.xlim(plt.xlim()[0], 1)
    # plt.xlim(1e-2, 1)
    # plt.legend(loc='upper left')
    plt.savefig(savename)

if __name__ == '__main__':
    plot_spectrum(*sys.argv[1:])