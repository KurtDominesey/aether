import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(filename, savename):
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    for name in table.dtype.names:
        marker = 'o'
        xdata = enrichments
        ydata = table[name]
        if ydata[0] == 0:
            ydata = ydata[1:]
            # plt.semilogy(xdata[1:], ydata, label=name, marker=marker, ls='--')
            xdata = xdata[:-1]
        plt.semilogy(xdata, ydata, label=name, marker=marker)
    # plt.xticks(refinements)
    plt.legend(loc='upper right')
    plt.title('L2 Convergence')
    plt.ylabel('L2 Error')
    plt.xlabel('Modes')
    plt.savefig(savename, transparent=True)

if __name__ == '__main__':
    filename = sys.argv[1]
    savename = sys.argv[2]
    plot_convergence(filename, savename)