import ast
import copy
import math
import sys
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import openmc

ALPHA = 0.6

JCP = True

def right_yticks():
    ax2 = plt.gca().secondary_yaxis('right')
    ax2.tick_params(axis='y', which='both', direction='in')
    plt.setp(ax2.get_yticklabels(), visible=False)

def plot_coarses(filename, savename, show_pgd=True, show_scalar=False,
                 **kwargs):
    if isinstance(show_pgd, str):
        show_pgd = ast.literal_eval(show_pgd)
    # groups = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']
    folder = '/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/'
    groups = np.load(folder+'casmo-sh-70.npy')[::-1]
    widths = groups[1:] - groups[:-1]
    widths = widths[::-1]
    table = np.genfromtxt(filename, names=True)
    alpha = ALPHA
    alpha2 = 0.6 #0.35
    coarsened = table['flux_coarsened'] / np.sqrt(widths)
    norm_coarsened = sum(coarsened**2)**0.5
    names = list(table.dtype.names)
    for i, name in enumerate(names):
        if 'norm' in name or 'source' in name:
            continue
        kwargs_line = copy.copy(kwargs)
        kwargs_line['alpha'] = alpha
        match = re.match('.*_m([0-9]+)$', name)
        if match:
            is_modal = True
            incr = int(match.group(1))
        else:
            is_modal = False
        if 'coarse' in name:
            if 'ip' in name and show_pgd:
                kwargs_line['ls'] = ':'
                # kwargs_line['alpha'] /= 2
            if 'sep' in name:
                kwargs_line['color'] = 'gray'
            elif is_modal:
                if not show_pgd:
                    continue
                # kwargs_line['ls'] = '-'
                incrs = (1, 10, 20, 30)
                colors = {incr: 'C' + str(i) for i, incr in enumerate(incrs)}
                if not incr in incrs:
                    continue
                kwargs_line['color'] = colors[incr]
                kwargs_line['label'] = 'Coarse, $M={}$'.format(incr)
                if incr == 1:
                    pass
                    # kwargs_line['ls'] = '-'
                    # kwargs_line['color'] = 'gray'
            else:
                kwargs_line['label'] = 'Coarse'
                if 'ip' in name:
                    kwargs_line['label'] = 'Coarse, IP'
                    kwargs_line['color'] = 'black' if show_pgd else 'C1'
                else:
                    kwargs_line['color'] = 'black' if show_pgd else 'C0'
                if i < len(table.dtype.names) - 1:
                    names.append(name)
                    continue
                # kwargs_line['color'] = None
        if 'flux' in name or 'pgd' in name or 'abs' in name or 'svd' in name:
            continue
        if '_m_' not in name and show_pgd:
            if show_scalar:
                continue
        if '_m_' in name:
            if show_pgd and not show_scalar:
                continue
            # kwargs_line['ls'] = '--'
            # kwargs_line['alpha'] = 0.5
            # kwargs_line['ls'] = '-.' if kwargs_line.get('ls', '') == '--' else ':'
            # kwargs_line['color'] = 'gray'
        # label = name.replace('_', r'\_')
        ydata = np.array(table[name])
        # ydata *= 1e2
        plt.step(groups[::-1], np.append(ydata[0], ydata), where='pre',
                 **kwargs_line)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Relative $L^2$ Error')
    plt.xlabel('Energy [eV]')
    plt.tight_layout(pad=0.2)
    # yticks for each decade
    locmaj = plt.gca().yaxis.get_major_locator()
    locs = locmaj()
    numdecs = math.log10(locs[-1] / locs[0])
    locmaj.set_params(numdecs=numdecs, numticks=numdecs)
    locmin = plt.gca().yaxis.get_minor_locator()
    locmin.set_params(subs=np.arange(1, 10), numdecs=numdecs, numticks=9*numdecs)
    # plt.legend(loc='lower left', ncol=2)
    # right_yticks()
    # locmaj_x = plt.gca().xaxis.get_major_locator()
    # locs_x = locmaj_x()
    # numdecs_x = math.log10(locs_x[-1] / locs_x[0])
    # # locmaj_x.set_params(numdecs=numdecs_x, numticks=numdecs_x)
    # locmin_x = plt.gca().xaxis.get_minor_locator()
    # locmin_x.set_params(subs=np.arange(1, 11), numdecs=numdecs_x, 
    #                     numticks=10*numdecs_x)
    plt.savefig(savename)

def line(**kwargs):
    return matplotlib.lines.Line2D([], [], **kwargs)

def plot_one(*args):
    plt.style.use('thesis.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 2.75))
    plot_coarses(*args)

def plot_all(*args):
    name = sys.argv[1]
    suffix = sys.argv[2]
    fuels = sys.argv[3:]
    fuels_str = '-'.join(str(fuel) for fuel in fuels)
    savename = name.format(fuel=fuels_str)+'_'+suffix+'.pdf'
    matplotlib.rc('lines', linewidth=1.25)
    quantities = ['Angular, $\\psi$', 'Scalar, $\\phi$']
    ncols = len(quantities)
    nrows = len(fuels)
    ij = 0
    for i, fuel in enumerate(fuels):
        for j, quantity in enumerate(quantities):
            ij += 1
            if j == 0:
                axij = plt.subplot(nrows, ncols, ij)
                axj0 = axij
            else:
                axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
            plot_coarses(name.format(fuel=fuel)+'.txt', savename, True, j)
            plt.xlabel(None)
            plt.ylabel(None)
            if j > 0:
                plt.setp(axij.get_yticklabels(), visible=False)
            if j == ncols - 1:
                axij.yaxis.set_label_position('right')
                fancy = {'uo2': r'UO\textsubscript{2}', 
                         'mox43': r'4.3\% MOX'}
                plt.ylabel(fancy[fuel])
            if i < nrows - 1:
                plt.setp(axij.get_xticklabels(), visible=False)
                plt.gca().xaxis.set_ticklabels([])
            else:
                pass
            if i == 0:
                plt.title(quantity)
                upper = 1.45 if not JCP else 1.525
                if j == 0:
                    lines = [line(color='black', ls=ls, label=label)
                             for label, ls in (('Consistent-P', '-'), 
                                               ('Inconsistent-P', ':'))]
                    plt.legend(handles=lines, loc='upper center',
                               title='Transport Correction', ncol=2,
                               bbox_to_anchor=(0.5, upper))
                if j == ncols - 1:
                    lines = [line(alpha=ALPHA, color='C'+str(i), label=label)
                             for i, label in enumerate((1, 10, 20, 30))]
                    plt.legend(handles=lines, loc='upper center', 
                               title='Cross-Sections, $M=$',
                               ncol=len(lines), bbox_to_anchor=(0.5, upper),
                               columnspacing=1.3)
            # handles = plt.gca().get_legend_handles_labels()
            if i == nrows - 1 and j == 0:
                lines = [line(alpha=ALPHA, color='black', 
                              label='Full-order cross-sections')]
                plt.legend(handles=lines, loc='lower left')
            if j == ncols - 1:
                plt.setp(axij.get_yticklabels(), visible=False)
                right_yticks()
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.0, 
                     rect=(0.03, 0.04, 1, 1))
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel('Relative $L^2$ Error', labelpad=32.5 if not JCP else 30)
    ax0.set_xlabel('Energy [eV]', labelpad=20 if not JCP else 17.5)
    # ax0.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.225),
    #            title='Transport Correction')
    plt.savefig(savename)

if __name__ == '__main__':
    # python plot_coarses.py Fuel_CathalauCoarseTestUniformFissionSource_{fuel} ip uo2 mox43
    if JCP:
        plt.style.use('jcp.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 4))
    else:
        plt.style.use('thesis.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 4.5))
    plot_all(*sys.argv[1:])