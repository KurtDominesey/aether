import sys
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_compare(filename, show_ip=False, **kwargs):
    # ymin = np.inf
    # ymax = -np.inf
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    fig = plt.gcf().number
    plt.figure()
    for name in table.dtype.names:
        if 'j2' not in name:
            continue
        if 'svd' in name:
            continue
        if show_ip and 'ip' not in name:
            continue
        if not show_ip and 'ip' in name:
            continue
        # table[name] *= 100
        plt.plot([0, 1], np.abs([table[name][1], table[name][-1]]))
    plt.yscale('log')
    ymin, ymax = plt.ylim()
    plt.clf()
    plt.figure(fig)
    for name in table.dtype.names:
        # if name != 'error_d':
        if 'j2' not in name:
            continue
        # plt.title('uo2')
        # if name[-1] == 'm':
        #     kwargs['color'] = plt.gca().lines[-1].get_color()
        #     kwargs['ls'] = '--'
        marker = '.' #'o'
        xdata = enrichments
        ydata = table[name] #/ table['error_d'][0]
        ydata = np.abs(ydata)
        if ydata[0] == 0:
            ydata = ydata[1:]
            xdata = xdata[:-1]
        # ydata /= ydata[0]
        if 'inf' in name:
            marker = None
            kwargs['ls'] = '--'
        elif 'svd' in name:
            kwargs['ls'] = ':'
        else:
            kwargs['ls'] = '-'
        g_name = ''
        for char in name.split('g')[-1]:
            if not char.isdigit():
                break
            g_name += char
        g = int(g_name)
        if g < 21 or g > 27:
            continue
        if 'svd' in name:
            continue
        if show_ip and 'ip' not in name:
            continue
        if not show_ip and 'ip' in name:
            continue
        color = 'C' + str((g % 10) - 1)
        # color = 'C' + str(27 - g)
        kwargs['color'] = color
        label = None if ('inf' in name or 'svd' in name) else str(g)
        plt.plot(xdata, ydata, label=label, marker=marker, **kwargs)
        # plt.plot(np.extract(table[name] > 0, xdata),
        #          np.extract(table[name] > 0, ydata), ls=None,
        #          marker='+', markersize=10)
    # plt.xticks(refinements)
    # plt.legend(loc='best', title='Group $g=$', ncol=2)
    # plt.ylim(1e-5, plt.ylim()[1])
    plt.yscale('log')
    logticks(plt.gca().yaxis)
    # plt.xlabel('Modes $M$')
    # plt.ylabel(r'Relative Error, ' +
    #            r'$\left\vert\Sigma_{t,g}-\Sigma_{t,g}^{\mathrm{PGD}}\right\vert' +
    #            r'/\Sigma_{t,g}$')
    # plt.savefig(savename)
    # plt.close()
    plt.gca().xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    return (ymin, ymax)

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

if __name__ == '__main__':
    # python plot_compare_mgxs.py Fuel_CathalauMgxsTestToCasmo70_{fuel} ip uo2 mox43

    # main(sys.argv[1])
    plt.style.use('./thesis.mplstyle')
    # matplotlib.rc('legend', fontsize=10.95, title_fontsize=10.95)
    # matplotlib.rc('axes.grid', which='both')
    # matplotlib.rc('ytick', right=True, direction='in')
    name = sys.argv[1]
    suffix = sys.argv[2]
    fuels = sys.argv[3:]
    # matplotlib.rc('figure', figsize=(6.5, 0.5+2.5*len(fuels)))
    order = 1
    nrows = order + 1
    ncols = len(fuels)
    # matplotlib.rc('figure', figsize=(6.5, (6+3/8)/2 * nrows))
    matplotlib.rc('figure', figsize=(6.5, 4.5))
    ij = 0
    for ell in range(order+1):
        for j, fuel in enumerate(fuels):
            i = ell
            # print(ell)
            ij += 1
            if j == 0:
                axij = plt.subplot(nrows, ncols, ij)
                axj0 = axij
                plt.autoscale(enable=False, axis='y')
            else:
                axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
            if ell == 0:
                dashed = matplotlib.lines.Line2D([], [], color='black', ls='--',
                    label='Homogenized')
                locs = {'mox43': 'lower left', 'uo2': 'lower left'}
                legend = plt.legend(handles=[dashed], loc=locs[fuel])
                plt.gca().add_artist(legend)
            if j > 0:
                ymin0, ymax0 = (ymin, ymax)
            ymin, ymax = plot_compare(name.format(fuel=fuel)+'.txt', ell, 
                                      markevery=2, alpha=0.8)
            if j > 0:
                ymin = min(ymin, ymin0)
                ymax = max(ymax, ymax0)
            plt.ylim(ymin, ymax)
            if j > 0:
                plt.setp(axij.get_yticklabels(), visible=False)
            else:
                pass
            if j == ncols - 1:
                axij.yaxis.set_label_position('right')
                # plt.ylabel(r'$\Sigma_{t,g,\ell=%i}$' % ell)
                plt.ylabel(r'$\ell=%i$' % ell)
            if i < nrows - 1:
                plt.setp(axij.get_xticklabels(), visible=False)
                plt.gca().xaxis.set_ticklabels([])
            else:
                pass
            if i == 0:
                fancy = {'uo2': r'UO\textsubscript{2}', 
                         'mox43': r'4.3\% MOX'}
                plt.title(fancy[fuel])
            handles = plt.gca().get_legend_handles_labels()
            if j == ncols - 1:
                plt.setp(axij.get_yticklabels(), visible=False)
                right_yticks()
                plt.setp(axij.get_yticklabels(), visible=False)
    # plt.gca().autoscale(True, axis='x', tight=True)
    # plt.tight_layout(pad=0.2)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                     rect=(0.05, 0.035, 1, 0.89))
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_ylabel(
        r'Relative Error, '
        r'$\left\vert\Sigma_{t,g,\ell}-\Sigma_{t,g,\ell}^{\mathrm{PGD}}\right\vert'
        r'/\Sigma_{t,g,\ell}$',
        labelpad=32.5)
    ax0.set_xlabel('Modes $M$', labelpad=20)
    ax0.legend(*handles, loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.225),
               title='Group $g=$')
    # legend = ax0.legend(reversed(handles), reversed(desc), loc='upper center', 
    #                     ncol=math.ceil(len(handles)/2),
    #                     bbox_to_anchor=(0.49, 1.14))
    fuels_str = '-'.join(str(fuel) for fuel in fuels)
    plt.savefig(name.format(fuel=fuels_str)+'_'+suffix+'.pdf')