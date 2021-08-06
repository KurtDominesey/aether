import math
import sys
import re

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import openmc

ALPHA = 0.75

JCP = True

# def right_yticks():
#     ax2 = plt.gca().secondary_yaxis('right')
#     ax2.tick_params(axis='y', which='both', direction='in')
#     plt.setp(ax2.get_yticklabels(), visible=False)

def plot_coarse(filename, savename, is_relative, show_scalar=False):
    # groups = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']
    folder = '/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/'
    groups = np.load(folder+'casmo-sh-70.npy')[::-1]
    # widths = groups[1:] - groups[:-1]
    widths = np.log(groups[1:]/groups[:-1])
    widths = widths[::-1]
    # widths = np.append(widths, widths[-1])
    table = np.genfromtxt(filename, names=True)
    alpha = 0.75
    alpha2 = 0.5
    # coarsened = table['flux_coarsened']
    coarsened = table['l2_norm_d']
    norm_coarsened = sum(coarsened**2 / widths)**0.5
    coarsened_m = table['l2_norm_m']
    norm_coarsened_m = sum(coarsened_m**2 / widths)**0.5
    print('norm_coarsened', norm_coarsened)
    print('norm_coarsened_m', norm_coarsened_m)
    is_sep = False
    for name in table.dtype.names:
        invisible = False
        if 'source' in name or 'norm' in name:
            continue
        kwargs_line = {'alpha': alpha}
        match = re.match('.*_m([0-9]+)$', name)
        if match:
            is_modal = True
            incr = int(match.group(1))
        else:
            is_modal = False
        if 'coarse' in name:
            if 'sep' in name:
                is_sep = True
                # kwargs_line['color'] = 'gray'
                kwargs_line['ls'] = ':'
            elif is_modal:
                continue
                kwargs_line['ls'] = '--'
                if not incr in (10, 20):
                    continue
            else:
                kwargs_line['color'] = 'black'
            if 'ip' in name:
                kwargs_line['color'] = 'gray'
                kwargs_line['zorder'] = 3
            else:
                pass
                kwargs_line['color'] = 'black'
                kwargs_line['zorder'] = 4
        if 'flux' in name:
            continue
        if 'svd' in name:
            continue
        if 'pgd' in name or 'svd' in name:
            # continue
            incr = name.split('_')[-1].split('m')[-1]
            incr = int(incr)
            # if not (incr / 10) % 2:
            #     continue
            if not incr <= 30:
                invisible = True
            else:
                kwargs_line['label'] = 'PGD, $M={}$'.format(incr)
                # if incr not in [20, 40, 60]:
                #     continue
                # kwargs_line['color'] = 'C' + str()
                if 'svd' in name:
                    kwargs_line['alpha'] = 0.25
                    kwargs_line['color'] = {
                        10: 'C0',
                        20: 'C1',
                        30: 'C2'
                    }[incr]
        if is_relative and 'abs' in name:
            continue
        elif not is_relative and 'rel' in name:
            continue
        if '_m_' in name:
            # kwargs_line['color'] = plt.gca().lines[-1].get_color()
            # kwargs_line['ls'] = ':'
            kwargs_line['label'] = None
            if not show_scalar:
                continue
        elif show_scalar:
            continue
        label = name.replace('_', r'\_')
        xdata = range(len(table[name]))
        ydata = np.array(table[name])
        if 'abs' in name:
            pass
            # ydata /= np.sqrt(widths)
        # mids = (groups[1:] + groups[:-1])/2
        if 'abs' in name:
            # ydata /= np.sqrt(widths)
            norm = sum(ydata**2 / widths)**0.5
            ydata /= widths
            # if name == 'coarse_d_abs':
            #     norm_coarse = norm
            norm /= norm_coarsened
            ydata /= norm_coarsened
            # ls = '--' if '_m_' in name else '--'
            ls = '-.' if kwargs_line.get('ls', '') == ':' else '--'
        if not invisible:
            plt.step(groups[::-1], np.append(ydata[0], ydata), where='pre',
                    **kwargs_line)
        if 'abs' in name:
            print('{:.2e}'.format(norm), name)
            if not invisible:
                plt.axhline(norm, color=plt.gca().lines[-1].get_color(), ls=ls,
                            alpha=alpha2)
    plt.xscale('log')
    plt.yscale('log')
    if not is_relative:
        plt.ylabel('Normalized $L^2$ Error')
    else:
        plt.ylabel('Relative $L^2$ Error')
    plt.xlabel('Energy [eV]')
    # kw_black = {'color': 'black', 'alpha': alpha}
    # kw_black2 = {'color': 'black', 'alpha': alpha2}
    # styles = [matplotlib.lines.Line2D([], [], label='Angular', **kw_black),
    #           matplotlib.lines.Line2D([], [], label='Scalar', ls=':', 
    #                                   **kw_black)]
    # if not is_relative:
    #     styles.append(matplotlib.lines.Line2D([], [], ls='--', 
    #         label='Total angular', **kw_black))
    #     styles.append(matplotlib.lines.Line2D([], [], ls='-.',
    #         label='Total scalar', **kw_black))
    # legend = plt.legend(handles=styles, loc='lower left', 
    #                     ncol=2)
    # plt.gca().add_artist(legend)
    # lines = [matplotlib.lines.Line2D([], [], color='black', alpha=alpha)]
    # labels = ['\\vspace{+1cm}Coarse']
    # if is_sep:
    #     lines.append(matplotlib.lines.Line2D([], [], color='gray', 
    #                  alpha=alpha))
    #     labels[0] += ', similar'
    #     labels.append('Coarse, dissimilar')
    #     blank = 2 if is_relative else 0
    #     lines.insert(blank, matplotlib.lines.Line2D([], [], ls=''))
    #     labels.insert(blank, '')
    # for i in range(3):
    #     lines.append(matplotlib.lines.Line2D([], [], color='C'+str(i), 
    #                  alpha=alpha))
    #     labels.append('PGD, $M={}$'.format(10+i*20))
    # loc = 'upper right' if is_relative else 'lower right'
    # plt.legend(lines, labels, loc=loc, ncol=2)
    # plt.legend(loc='best')
    plt.tight_layout(pad=0.2)
    logticks(plt.gca().yaxis)
    # right_yticks()
    plt.savefig(savename)
    return is_sep

def plot_one(*args):
    plt.style.use('thesis.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 3.25))
    plot_coarse(*args)

def line(**kwargs):
    return matplotlib.lines.Line2D([], [], **kwargs)

def right_yticks():
    # ax2 = plt.gca().secondary_yaxis('right')
    ax = plt.gca()
    ax2 = ax.twinx()
    plt.yscale('log')
    plt.ylim(ax.get_ylim())
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator())
    ax2.yaxis.set_minor_locator(matplotlib.ticker.LogLocator())
    logticks(ax2.yaxis)
    plt.grid(visible=False)
    ax2.tick_params(axis='y', which='both', direction='in')
    plt.setp(ax2.get_yticklabels(), visible=False)

def logticks(axis):
    locmaj = axis.get_major_locator()
    locmin = axis.get_minor_locator()
    locs = locmaj()
    numdecs = math.log10(locs[-1]/locs[0])
    locmaj.set_params(numdecs=numdecs, numticks=numdecs)
    subs = np.arange(1, 10)
    locmin.set_params(subs=subs, numdecs=numdecs, numticks=len(subs)*numdecs)

def plot_all(name, suffix):
    savename = name + '_' + suffix + '.pdf'
    matplotlib.rc('lines', linewidth=1.25)
    quantities = ['Angular, $\\psi$', 'Scalar, $\\phi$']
    ylabels = ['Relative $L^2$ Error', 'Normalized $L^2$ Error [1/$u$]']
    ncols = len(quantities)
    nrows = len(ylabels)
    ij = 0
    for i, ylabel in enumerate(ylabels):
        for j, quantity in enumerate(quantities):
            ij += 1
            if j == 0:
                axij = plt.subplot(nrows, ncols, ij)
                axj0 = axij
            else:
                axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
            is_sep = plot_coarse(name+'.txt', savename, not i, j)
            plt.xlabel(None)
            ylabel = plt.ylabel(ylabel if j == 0 else None)
            if j == 0 and JCP:
                # push ylabel down to avoid whitespace between subplots
                x, y = ylabel.get_position()
                plt.gca().get_yaxis().set_label_coords(
                    x, y*0.885, transform=ylabel.get_transform())
            if j > 0:
                plt.setp(axij.get_yticklabels(), visible=False)
            if i < nrows - 1:
                plt.setp(axij.get_xticklabels(), visible=False)
                plt.gca().xaxis.set_ticklabels([])
            else:
                pass
            if i == 0:
                plt.title(quantity)
                upper = 1.45 if not JCP else 1.525
                if j == 0:
                    lines = [line(color=color, label=label)
                             for label, color in (('Consistent-P', 'black'), 
                                                  ('Inconsistent-P', 'gray'))]
                    plt.legend(handles=lines, loc='upper center',
                               title='Transport Correction', ncol=2,
                               bbox_to_anchor=(0.5, upper))
                if j == ncols - 1:
                    lines = [line(alpha=ALPHA, color='C'+str(i), label=label)
                             for i, label in enumerate((10, 20, 30))]
                    plt.legend(handles=lines, loc='upper center', 
                               title='PGD, $M=$',
                               ncol=len(lines), bbox_to_anchor=(0.5, upper))
            # handles = plt.gca().get_legend_handles_labels()
            if i == nrows - 1 and j == 0:
                if not is_sep:
                    lines = [line(alpha=ALPHA, color='black', 
                                label='Full-order coarse group')]
                    ncol = 1
                else:
                    lines = [line(alpha=ALPHA, color='black', 
                                  ls=ls, label=label)
                             for label, ls in (('Similar', '-'),
                                               ('Dissimilar', ':'))]
                    ncol = 2
                plt.legend(handles=lines, loc='lower left', ncol=ncol)
            if j == ncols -1 and i == nrows - 1:
                lines = [line(alpha=ALPHA, color='black', ls='--',
                             label='Total error')]
                plt.legend(handles=lines, loc='lower left')
            if j == ncols - 1:
                plt.setp(axij.get_yticklabels(), visible=False)
                right_yticks()
                plt.setp(axij.get_yticklabels(), visible=False)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=-0.1, 
                     rect=(0, 0.04, 1, 1 if not JCP else 1))
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Energy [eV]', labelpad=20 if not JCP else 18)
    plt.savefig(savename)

if __name__ == '__main__':
    if JCP:
        plt.style.use('jcp.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 4))
    else:
        plt.style.use('thesis.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 4.5))
    plot_all(*sys.argv[1:])
    # python plot_coarse.py Fuel_CathalauCoarseTestUniformFissionSource_uo2 pgd
    # python plot_coarse.py Fuel_CathalauCoarseTestUniformFissionSource_mox43 pgd
    # python plot_coarse.py Pattern_CheckerboardTestUnequalPowers_0 jcp
    # python plot_coarse.py CheckerboardTestUniformFissionSource pgd