import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_convergence(filename, savename):
    table = np.genfromtxt(
        filename, skip_header=1, missing_values=['-'], filling_values=[0.0])
    num_rows = len(table[:, 0])
    num_cols = len(table[0, :])
    refinements = range(num_rows)
    for i in range(int(num_cols)):
        label = 'g' + str(i)
        marker = '.'
        plt.semilogy(refinements, table[:, i], label=label, marker=marker)
    plt.xticks(refinements)
    plt.legend(loc='upper right')
    plt.title('L2 Convergence')
    plt.ylabel('L2 Error')
    plt.xlabel('Mesh Refinements')
    plt.savefig(savename, transparent=True)

if __name__ == '__main__':
    filename = sys.argv[1]
    savename = sys.argv[2]
    plot_convergence(filename, savename)