import ast

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import openmc


GROUP_STRUCTURE = 'CASMO-70'


def plot_spectrum(adjoint):
    print(adjoint)
    filename = f'mox43_{GROUP_STRUCTURE}_spectrum{"_adj" if adjoint else ""}'
    table = np.genfromtxt(filename+'.txt')
    xdata = openmc.mgxs.GROUP_STRUCTURES[GROUP_STRUCTURE][::-1]
    widths = openmc.mgxs.GROUP_STRUCTURES[GROUP_STRUCTURE]
    # widths = np.subtract(widths[1:], widths[:-1])
    widths = np.log(np.divide(widths[1:], widths[:-1]))
    widths = widths[::-1]
    print(widths)
    plt.xscale('log')
    ydata = table
    plt.step(xdata, np.append(ydata[0], ydata), where='pre')
    plt.savefig(filename+'.pdf')


if __name__ == '__main__':
    import sys
    plot_spectrum(*[ast.literal_eval(arg) for arg in sys.argv[1:]])