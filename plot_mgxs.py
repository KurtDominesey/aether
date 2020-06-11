import sys

import h5py
import matplotlib.pyplot as plt

import openmc.mgxs

def plot_mgxs(filename, names=None):
    file = h5py.File(filename, 'r')
    num_groups = file.attrs['energy_groups']
    if 'group structure' in file.attrs:
        group_structure = file.attrs['group structure']
        xdata = group_structure[::-1][:-1]
        plt.xscale('log')
    else:
        xdata = range(num_groups)
    if names is None:
        names = file.keys()
    for name in names:
        if name == 'void':
            continue
        material = file[name]
        plt.step(xdata, material['294K']['total'], where='mid', label=name)
        # if material.attrs['fissionable']:
        #     plt.step(xdata, material['294K']['chi'], where='mid', 
        #              label=name+' $\\chi$')
    # groups_coarse = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']
    # for g_rev, bound in enumerate(groups_coarse):
    #     g = (len(groups_coarse)-1) - g_rev
    #     # if bound > 3.5 and bound < 400:
    #     plt.axvline(bound, ls='--', color='black')
    # plt.xlim(plt.xlim()[0], 400)
    plt.yscale('log')

if __name__ == '__main__':
    plot_mgxs(sys.argv[1])
    plt.legend(loc='best')
    plt.savefig(sys.argv[2])
  