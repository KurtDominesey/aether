import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import openmc


def minor_yticks():
    locmaj = plt.gca().yaxis.get_major_locator()
    locs = locmaj()
    numdecs = math.log10(locs[-1] / locs[0])
    locmaj.set_params(numdecs=numdecs, numticks=numdecs)
    locmin = plt.gca().yaxis.get_minor_locator()
    locmin.set_params(subs=np.arange(1, 10), numdecs=numdecs, 
                      numticks=9*numdecs)


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
    plt.sca(ax)


def plot_groups(filename, structure, yfloor):
    table = np.genfromtxt(filename)
    # table = table[::2, :]
    mm, num_groups = table.shape
    groups = openmc.mgxs.GROUP_STRUCTURES[structure]
    assert num_groups == len(groups) - 1
    widths = np.log(groups[1:]/groups[:-1])
    widths = widths[::-1]
    for m in range(mm):
        ydata = table[m, :]
        total = sum(ydata**2 / widths)**0.5
        ydata /= widths
        color = f'C{m}'
        plt.step(groups[::-1], np.append(ydata[0], ydata), where='pre',
                 color=color, alpha=0.8, label=str((m+1)*10))
        plt.axhline(total, color=color, ls='--', alpha=0.5)
        print('{:.2e}'.format(total))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1.1e-3, plt.xlim()[1])
    if yfloor is not None:
        ymin, ymax = plt.ylim()
        plt.ylim(max(yfloor, ymin), ymax)
    minor_yticks()
    return plt.ylim()


def plot_fuel(base, savename, yfloors):
    fluxes = ['d', 'm']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    num_rows = len(structures)
    num_cols = len(fluxes)
    i = 0
    for row, structure in enumerate(structures):
        ymin_row, ymax_row = (np.nan, np.nan)
        axes_row = []
        for col, flux in enumerate(fluxes):
            i += 1
            plt.subplot(num_rows, num_cols, i)
            axes_row.append(plt.gca())
            filename = base.format(structure=structure, flux=flux)
            print(fuel, structure, flux, algorithm)
            ymin, ymax = plot_groups(filename, structure, yfloors[row])
            ymin_row = min(ymin, ymin_row)
            ymax_row = max(ymax, ymax_row)
            if row == 0:
                fancy = {'d': 'Angular, $\\psi$', 'm': 'Scalar, $\\phi$'}
                plt.title(fancy[flux])
            if col == num_cols - 1:
                plt.gca().yaxis.set_label_position('right')
                plt.ylabel(structure)
            if row < num_rows - 1:
                plt.setp(plt.gca().get_xticklabels(), visible=False)
            if col > 0:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
                right_yticks()
                plt.setp(plt.gca().get_yticklabels(), visible=False)
            if row == num_rows -1 and col == 0:
                line = matplotlib.lines.Line2D([], [], ls='--', alpha=0.5,
                                               color='k', label='Total error')
                plt.legend(handles=[line], loc='lower left')
        for ax in axes_row:
            ax.set_ylim(ymin_row, ymax_row)
    handles, labels = plt.gca().get_legend_handles_labels()
    rect = [0.03, 0.025, 1, 0.915]
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, rect=rect)
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Energy [eV]', labelpad=17.5)
    ax0.set_ylabel('Normalized $L^2$ Error [1/lethargy]', labelpad=30)
    bbox_to_anchor = [0.5, 1.165]
    ax0.legend(handles, labels, loc='upper center', ncol=len(handles), 
               bbox_to_anchor=bbox_to_anchor, title='Modes $M$=',
               columnspacing=3)
    plt.savefig(savename)
    plt.close()


if __name__ == '__main__':
    # python plot_compare_groupwise.py
    plt.style.use('jcp.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 5.5))
    matplotlib.rc('legend', fontsize=10)
    algorithms = ['svd', 'pgd']
    fuels = ['uo2', 'mox43']
    base = 'GroupStructure_CathalauCompareTestWithUpdate_{fuel}_{{structure}}' \
            '_{algorithm}_{{flux}}_groupwise.txt'
    savename = 'compare-groupwise-{fuel}-{algorithm}.pdf'
    yfloors_pgd = [0, 5e-7, 1e-6]
    yfloors_svd = [2e-10, 3e-8, 1e-7]
    for algorithm in algorithms:
        for fuel in fuels:
            plot_fuel(base.format(fuel=fuel, algorithm=algorithm), 
                      savename.format(fuel=fuel, algorithm=algorithm),
                      yfloors_pgd if algorithm == 'pgd' else yfloors_svd)