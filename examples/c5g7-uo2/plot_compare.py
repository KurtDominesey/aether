import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_compare(filename, savename):
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    names = list(table.dtype.names)
    ii = len(names)
    for i, name in enumerate(names):
        marker = '.' #'o'
        xdata = enrichments
        ydata = np.array(table[name])
        ydata /= table['error_d'][0]
        # if ydata[0] == 0:
        #     ydata = ydata[1:]
        #     xdata = xdata[:-1]
        # ydata /= ydata[0]
        if 'svd' in name and i < ii:
            names.append(name)
            continue
        ylim = plt.ylim()
        labels = {'error_d': 'Error $\\psi$, PGD',
                  'error_m': 'Error $\\phi$, PGD',
                  'residual': 'Residual',
                  'residual_streamed': 'Residual, streamed',
                  'residual_swept': 'Residual, swept',
                  'error_svd_d': 'Error $\\psi$, SVD',
                  'error_svd_m': 'Error $\\phi$, SVD',
                  'norm': 'Mode $M-1$'}
        ls = '-'
        if 'svd' in name:
            color = 'C3'
        elif 'error' in name:
            color = 'C0'
        if 'residual' in name:
            color = 'C1'
            if 'streamed' in name:
                ls = '--'
            elif 'swept' in name:
                ls = ':'
        if '_m' in name:
            ls = '--'
        if 'norm' in name:
            color = 'C2'
        plt.semilogy(xdata, ydata, label=labels[name], color=color, ls=ls, 
                     marker=marker, alpha=1.)
        if i + 1 == ii:
            plt.ylim(ylim)
            # plt.autoscale(False, 'y')
    # plt.xticks(refinements)
    legend = plt.legend(loc='upper right', ncol=2)  # title='Error / Error Indicator'
    # plt.title('L2 Convergence')
    plt.ylabel('Relative $L2$ Norm')
    plt.xlabel('Modes $M$')
    plt.tight_layout(pad=0.2)
    plt.savefig(savename)
    plt.close()

def main(ext, suffix=''):
    name_base = 'C5G7CompareTest{mode}'
    modes = ('Progressive', 'WithUpdate')
    for mode in modes:
        name = name_base.format(mode=mode)
        plot_compare(name+'.txt', name+suffix+'.'+ext)

if __name__ == '__main__':
    small = 10.95
    footnotesize = 10
    import mpl_rc
    mpl_rc.set_rc(small)
    matplotlib.rc('figure', figsize=(6.5, 2+3/8))  # 2.625
    matplotlib.rc('legend', fontsize=footnotesize)
    matplotlib.rc('legend', title_fontsize=small)
    # matplotlib.rc('font', family='sans-serif', serif=['Helvetica'])
    # matplotlib.rc('grid', linestyle=':')
    # matplotlib.rc('ytick', right=True, direction='in')
    # matplotlib.rc('xtick', top=True, direction='in')
    main(*sys.argv[1:])