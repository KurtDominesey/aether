import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_coarse(filename, savename):
    table = np.genfromtxt(filename, names=True)
    for name in table.dtype.names:
        if 'flux' in name or 'rel' in name:
            continue
        xdata = range(len(table[name]))
        plt.step(xdata, table[name], where='mid', label=name, alpha=0.75)
        if 'abs' in name:
            norm = sum(table[name]**2)**0.5
            plt.axhline(norm, color=plt.gca().lines[-1].get_color(), ls=':')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(savename)

if __name__ == '__main__':
    plot_coarse(*sys.argv[1:])