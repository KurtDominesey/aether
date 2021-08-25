import collections
import math
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


JCP = True  # J. Comput. Phys. style plot


def right_yticks():
    ax = plt.gca()
    ax2 = plt.gca().twinx()
    plt.yscale('log')
    plt.ylim(ax.get_ylim())
    locmaj = ax.yaxis.get_major_locator()
    numdecs = math.log10(locmaj()[-1]/locmaj()[0])
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator())
    ax2.yaxis.get_major_locator().set_params(numdecs=numdecs, numticks=numdecs)
    subs = np.arange(1, 10)
    ax2.yaxis.set_minor_locator(matplotlib.ticker.LogLocator())
    ax2.yaxis.get_minor_locator().set_params(subs=subs, numdecs=numdecs,
                                             numticks=len(subs)*numdecs)
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    plt.grid(visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)


def plot_compare(filebase, flux, **kwargs):
    projs = ('', 'Minimax')  # '' is Galerkin
    algorithms = ('Progressive', 'WithUpdate',)
    first = True
    for i, proj in enumerate(projs):
        for j, alg in enumerate(algorithms):
            ij = i * len(algorithms) + j
            # if proj == '' and alg == 'Progressive':
            #     continue
            filename = filebase.format(proj=proj, algorithm=alg)
            table = np.genfromtxt(filename, names=True)
            enrichments = list(range(table.shape[0]))
            plots = ['error_{flux}'.format(flux=flux)]
            if first:
                plots = ['error_svd_{flux}'.format(flux=flux)] + plots
            for name in plots:
                kwargs_line = kwargs
                kwargs_line['color'] = 'C' + str(i)
                if 'svd' in name:
                    last_color =  2
                    kwargs_line['color'] = 'C' + str(last_color)
                    kwargs_line['label'] = 'SVD'
                else:
                    label = proj + (' ' if proj else '')
                    if 'Update' in alg:
                        label += 'Update'
                    else:
                        label += 'Prog.'
                    kwargs_line['label'] = label
                kwargs_line['ls'] = '-'
                if 'Update' in alg or 'svd' in name:
                    kwargs_line['alpha'] = 0.8
                    kwargs_line['marker'] = 'o'
                else:
                    kwargs_line['alpha'] = 0.5
                    kwargs_line['marker'] = 'D'
                xdata = enrichments
                ydata = table[name] / table[name][0]
                ydata = np.abs(ydata)
                plt.plot(xdata, ydata, **kwargs_line)
            first = False
    plt.yscale('log')
    plt.tight_layout(pad=0.2)
    locmaj = plt.gca().yaxis.get_major_locator()
    locs = locmaj()
    numdecs = math.log10(locs[-1] / locs[0])
    locmaj.set_params(numdecs=numdecs, numticks=numdecs)
    locmin = plt.gca().yaxis.get_minor_locator()
    locmin.set_params(subs=np.arange(1, 10), numdecs=numdecs, 
                      numticks=9*numdecs)
 

def main():
    name_base = 'GroupStructure_CathalauCompareTest{{proj}}{{algorithm}}' \
                 '_{fuel}_{structure}.txt'
    fuels = ['uo2', 'mox43']
    fluxes = ['d', 'm']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    nrows = len(structures)
    ncols = len(fuels)
    for fuel in fuels:
        ij = 0
        for i, structure in enumerate(structures):
            for j, flux in enumerate(fluxes):
                ij += 1
                if ij == 1:
                    axij = plt.subplot(nrows, ncols, ij)
                    axj0 = axij
                elif j == 0:
                    axij = plt.subplot(nrows, ncols, ij)
                    axj0 = axij
                else:
                    axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
                name = name_base.format(structure=structure, fuel=fuel)
                plt.gca().set_prop_cycle(None)
                plot_compare(name, flux, markevery=2, markersize=2.75)
                handles, desc = plt.gca().get_legend_handles_labels()
                plt.gca().tick_params(axis='y', which='both', left=True)
                if j > 0:
                    plt.setp(axij.get_yticklabels(), visible=False)
                if j == ncols - 1:
                    axij.yaxis.set_label_position('right')
                    plt.ylabel(structure)
                if i < nrows - 1:
                    plt.setp(axij.get_xticklabels(), visible=False)
                    plt.gca().xaxis.set_ticklabels([])
                if i == 0:
                    fancy = {'d': r'Angular, $\psi$', 
                             'm': r'Scalar, $\phi$'}
                    plt.title(fancy[flux])
                if j == ncols - 1:
                    right_yticks()
                    plt.setp(axij.get_yticklabels(), visible=False)
        rect = [0.03, 0.025, 1, 0.925]
        if JCP:
            rect[-1] = 0.935
        plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                        rect=rect)
        ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
        ax0.set_xticks([])
        ax0.set_yticks([])
        ax0.set_xlabel('Modes $M$', labelpad=20 if not JCP else 17.5)
        ax0.set_ylabel('Normalized $L^2$ Error', labelpad=32.5 if not JCP else 30)
        bbox_to_anchor = [0.49, 1.14]
        if JCP:
            bbox_to_anchor[1] = 1.145
        legend = ax0.legend(handles, desc, loc='upper center', 
                            ncol=len(handles), columnspacing=1.9,
                            bbox_to_anchor=bbox_to_anchor)
        plt.savefig('compare-minimax-{fuel}.pdf'.format(fuel=fuel))
        plt.close()


if __name__ == '__main__':
    # python plot_compare_minimax.py
    plt.style.use('jcp.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 4.75))
    matplotlib.rc('legend', fontsize=10)
    main(*sys.argv[1:])