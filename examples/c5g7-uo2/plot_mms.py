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

if __name__ == '__main__':
    name_base = 'FEDegree_C5G7MmsOrderTest{rom}_{{degree}}.txt'
    plot_rate(name_base.format(rom='FullOrder'), range(0, 3), True, 
              ls='--', marker='o', alpha=0.75)
    plot_rate(name_base.format(rom='Pgd'), range(0, 3), False, 
              ls=':', marker='s', alpha=0.75)
    plt.savefig(sys.argv[1])