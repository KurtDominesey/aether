import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_rate(name_base, degrees, suffix, plot_theoretical=True, **kwargs):
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
        label = str(degree) + ', ' + suffix
        plt.plot(range(num_cycles), total, label=label, 
                 color='C'+str(degree % 10), **kwargs)
    plt.xticks(ticks=range(num_cycles))
    plt.yscale('log')
    legend = plt.legend(title='{Polynomial Order} $p=$', 
                        loc='lower left', framealpha=0.85, ncol=2)
    plt.xlabel('Mesh Refinements')
    plt.ylabel('$L2$ Error')

def plot_mms(savename):
    name_base = 'FEDegree_C5G7MmsTest{rom}_{{degree}}.txt'
    degrees = range(0, 3)
    black_line = matplotlib.lines.Line2D([], [], color='black', marker='*',
                                         label='$p+1$ convergence')
    legend = plt.legend(handles=[black_line], loc='upper right')
    plt.gca().add_artist(legend)
    plot_rate(name_base.format(rom='FullOrder'), degrees, 'PGD',
              True, ls='--', marker='o', alpha=0.75)
    plot_rate(name_base.format(rom='Pgd'), degrees, 'full-order',
              False, ls=':', marker='s', alpha=0.75)
    plt.tight_layout(pad=0.2)
    plt.savefig(savename)

if __name__ == '__main__':
    import mpl_rc
    # fontsize = 12
    small = 10.95
    footnotesize = 10
    figsize = (6.5, 2+3/8)
    mpl_rc.set_rc(small, figsize)
    matplotlib.rc('legend', fontsize=footnotesize)
    matplotlib.rc('legend', title_fontsize=footnotesize)
    plot_mms(sys.argv[1])
    print(sys.argv[1])