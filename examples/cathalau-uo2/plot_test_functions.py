import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import openmc

STRUCTURE = 'CASMO-70'

def plot_test_functions(filename):
    table = np.genfromtxt(filename+'.txt')
    num_functions, num_groups = table.shape
    groups = openmc.mgxs.GROUP_STRUCTURES[STRUCTURE]
    for func in range(min(10, num_functions)):
        ydata = table[func, :]
        assert len(ydata) == num_groups
        plt.step(groups[::-1], np.append(ydata[0], ydata), where='pre', 
                 label=str(func))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.savefig(filename+'.pdf')

if __name__ == '__main__':
    plot_test_functions('adjoint_test_functions')