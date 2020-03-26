import sys

import h5py
import matplotlib.pyplot as plt

def plot_mgxs(filename, savename):
    file = h5py.File(filename, 'r')
    num_groups = file.attrs['energy_groups']
    if 'group structure' in file.attrs:
        group_structure = file.attrs['group structure']
        xdata = group_structure[::-1][:-1]
    else:
        xdata = range(num_groups)
    for name, material in file.items():
        if name == 'void':
            continue
        plt.step(xdata, material['294K']['total'], where='mid', label=name)
    # plt.xscale('log')
    plt.legend(loc='best')
    plt.savefig(savename)

if __name__ == '__main__':
    plot_mgxs(sys.argv[1], sys.argv[2])
  