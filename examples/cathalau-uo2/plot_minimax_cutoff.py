import collections
import math
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


JCP = True  # J. Comput. Phys. style plot


def right_yticks():
    ax = plt.gca()
    ax2 = plt.gca().twinx()
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    plt.grid(visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)


def plot_minimax_ratio(filebase, **kwargs):
    projs = ('', 'Minimax')  # '' is Galerkin
    algorithms = ('Progressive', 'WithUpdate',)
    for j, alg in enumerate(algorithms):
        for i, proj in enumerate(projs):
            filename = filebase.format(proj=proj, algorithm=alg)
            table = np.genfromtxt(filename, names=True)
            if i == 0:
                assert proj == ''
                table_galerkin = table
            else:
                enrichments = list(range(table.shape[0]))
                for flux in ('d', 'm'):
                    name = 'error_' + flux
                    kwargs_line = kwargs
                    kwargs_line['color'] = 'C' + str(j)
                    label = {'Progressive': 'Prog.', 
                             'WithUpdate': 'Update'}[alg]
                    label += ', '
                    label += {'d': 'Angular', 'm': 'Scalar'}[flux]
                    kwargs_line['label'] = label
                    kwargs_line['ls'] = '-'
                    if flux == 'd':
                        kwargs_line['alpha'] = 0.8
                        kwargs_line['marker'] = 'o'
                    else:
                        kwargs_line['alpha'] = 0.5
                        kwargs_line['marker'] = 'D'
                    xdata = enrichments
                    ydata = [np.nan] * len(table[name])
                    for i, err in enumerate(table[name]):
                        for m, err_galerkin in enumerate(table_galerkin[name]):
                            if err_galerkin <= err:
                                ydata[i] = m - i
                                break
                    # print(ydata)
                    if 'Update' in alg:
                        print(ydata[30])
                    plt.plot(xdata, ydata, **kwargs_line)
    plt.tight_layout(pad=0.2)
    pad = -plt.xlim()[0]
    plt.xlim(plt.xlim()[0], xdata[-1]+pad)
 

def main():
    name_base = 'GroupStructure_CathalauCompareTest{{proj}}{{algorithm}}' \
                 '_{fuel}_{structure}.txt'
    fuels = ['uo2', 'mox43']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    nrows = len(structures)
    ncols = len(fuels)
    ij = 0
    for i, structure in enumerate(structures):
        for j, fuel in enumerate(fuels):
            print(fuel, structure)
            ij += 1
            axij = plt.subplot(nrows, ncols, ij)
            name = name_base.format(structure=structure, fuel=fuel)
            plt.gca().set_prop_cycle(None)
            plot_minimax_ratio(name, markevery=2, markersize=2.75)
            handles, desc = plt.gca().get_legend_handles_labels()
            axij.yaxis.set_major_locator(MultipleLocator(base=5))
            if j == ncols - 1:
                axij.yaxis.set_label_position('right')
                plt.ylabel(structure)
            if i == 0:
                fancy = {'uo2': r'UO\textsubscript{2}', 
                         'mox43': r'4.3\% MOX'}
                plt.title(fancy[fuel])
            if j == ncols - 1:
                pass
    rect = [0.03, 0.025, 1, 0.925]
    if JCP:
        rect[-1] = 0.935
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                    rect=rect)
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Modes $M$', labelpad=20 if not JCP else 17.5)
    ax0.set_ylabel("Number of modes obviated by Minimax PGD", 
                    labelpad=22)
    bbox_to_anchor = [0.49, 1.14]
    if JCP:
        bbox_to_anchor[1] = 1.145
    legend = ax0.legend(handles, desc, loc='upper center', 
                        ncol=len(handles), columnspacing=1.9,
                        bbox_to_anchor=bbox_to_anchor)
    plt.savefig('compare-minimax-cutoff.pdf')
    plt.close()


if __name__ == '__main__':
    # python plot_minimax_cutoff.py
    plt.style.use('jcp.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 4.75))
    matplotlib.rc('legend', fontsize=10)
    main(*sys.argv[1:])