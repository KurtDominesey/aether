import collections
import math
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_errors_k(fuel, structure, algo):
    filename = f'GroupStructure_CathalauCompareTest{algo}_{fuel}_{structure}.txt'
    table = np.genfromtxt(filename, names=True)
    return table['error_k'] * 1e-5


def line(**kwargs):
    return mpl.lines.Line2D([], [], **kwargs)

if __name__ == '__main__':
    # python plot_compare_k.py
    plt.style.use('jcp.mplstyle')
    mpl.rc('legend', fontsize=10)
    mpl.rc('figure', figsize=(6.5, 3.25))
    k_openmc = (1.326655146, 1.144720147)
    k_aether = [[1.323035211732793, 1.3212690674438736, 1.3285440989356505],
                [1.1440867037143927, 1.142720996232362, 1.148731731124236]]
    fuels = ['uo2', 'mox43']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    algorithms = ['WithEigenUpdate', 'MinimaxWithEigenUpdate']
    styles = ['--', '-']
    alpha = 0.8
    # the default matplotlib ylims
    ylims = [[1.2672371721241829, 1.339882389694907],
             [1.0329166950950899, 1.1824068365561489]]
    # but we want the y-axes of each subplot to span an equal interval
    ylens = [ylim[1]-ylim[0] for ylim in ylims]
    ylen_max = max(ylens)
    heights = [ylen/ylen_max for ylen in ylens]
    gs_kw = dict(height_ratios=heights)
    fig, axes = plt.subplots(2, 1, gridspec_kw=gs_kw)
    for i, fuel in enumerate(fuels):
        print('OpenMC', fuel, ':', '%.6f' % k_openmc[i])
        # plt.subplot(2, 1, i+1, gridspec_kw=gs_kw)
        plt.sca(axes[i])
        plt.grid(True, which='both', axis='both')
        # if not i:
        #     plt.gca().set_xticklabels([])
            # handles_color = [line(color='k', label='OpenMC')]
            # for j, structure in enumerate(structures):
            #     handles_color.append(line(color=f'C{j}', label=structure))
            # legend = plt.legend(handles=handles_color, ncol=2)
            # plt.gca().add_artist(legend)
        plt.axhline(k_openmc[i], color='k', alpha=alpha)
        for j, structure in enumerate(structures):
            color = f'C{j}'
            plt.axhline(k_aether[i][j], ls=':', color=color, alpha=alpha)
            print(structure, ':', '%.6f' % k_aether[i][j], 
                  '%.1f' % ((k_aether[i][j]-k_openmc[i])*1e5), 'pcm')
            for k, algo in enumerate(algorithms):
                k_pgd = k_aether[i][j] + get_errors_k(fuel, structure, algo)
                plt.plot(k_pgd, color=color, ls=styles[k], alpha=alpha)
        print(plt.ylim())
    handles = [line(color=f'C{i}', label=s) for (i, s) in enumerate(structures)]
    handles += [
        line(color='k', ls=':', label='Full-order model'),
        line(color='k', ls='--', label='Galerkin PGD'),
        line(color='k', ls='-', label='Minimax PGD')
    ]
    plt.legend(handles=handles, ncol=2)
    plt.xlabel('Modes $M$')
    # hide the spines
    axes[0].spines.bottom.set_visible(False)
    axes[1].spines.top.set_visible(False)
    axes[0].set_xticklabels([])
    axes[0].tick_params(axis='x', length=0)
    # add the line breaks
    d = -.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(1, d), (-1, -d)], markersize=12,
                linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
    axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)
    rect = [0.03, 0, 1, 1]
    plt.tight_layout(pad=0.25, rect=rect)
    # point out uo2 / mox43
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.text(
        0.35, 0.68, "UO\\textsubscript{2}", ha="center", va="center", rotation=90, size=15,
        transform=ax0.transAxes,
        bbox=dict(boxstyle="rarrow,pad=0.3", fc="w", ec="lightgray")
    )
    ax0.text(
        0.65, 0.68, "MOX", ha="center", va="center", rotation=90, size=15,
        transform=ax0.transAxes,
        bbox=dict(boxstyle="larrow,pad=0.3", fc="w", ec="lightgray")
    )
    ax0.set_ylabel('$k$-Eigenvalue', labelpad=35)
    plt.savefig('plot_compare_k-3.pdf')