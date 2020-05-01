import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_rate(name_base, degrees, plot_theoretical=True, **kwargs):
    for degree in degrees:
        name = name_base.format(degree=degree)
        table = np.genfromtxt(name, names=True)
        # for field in table.dtype.names:
        #     ydata = table[field]
        #     plt.plot(range(len(ydata)), ydata, label=field, marker='o')
        total = table['total']
        num_cycles = len(total)
        if plot_theoretical:
            theoretical = np.copy(total)
            for cycle in range(num_cycles-1):
                diff = num_cycles - (cycle + 1)
                factor = (2 ** diff) ** (degree + 1)
                theoretical[cycle] = factor * theoretical[-1]
            plt.plot(range(num_cycles), theoretical, 
                    marker='*', ls='-', color='black')
        if 'marker' not in kwargs:
            kwargs['marker'] = 'o'
        plt.plot(range(num_cycles), total, label=str(degree), 
                 color='C'+str(degree % 10), **kwargs)
    plt.yscale('log')
    plt.legend(loc='lower left')

def plot_mms(savename):
    name_base = 'FEDegree_CathalauMmsTest{rom}_{{degree}}.txt'
    degrees = range(0, 3)
    plot_rate(name_base.format(rom='FullOrder'), degrees, 
              True, ls='--', marker='o', alpha=0.75)
    plot_rate(name_base.format(rom='Pgd'), degrees, 
              False, ls=':', marker='s', alpha=0.75)
    plt.savefig(savename)

if __name__ == '__main__':
    plot_mms(sys.argv[1])