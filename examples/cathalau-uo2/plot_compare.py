import collections
import math
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

JCP = None

def right_yticks():
    ax = plt.gca()
    # ax.minorticks_on()
    # ax2 = plt.gca().secondary_yaxis('right')
    ax2 = plt.gca().twinx()
    plt.yscale('log')
    plt.ylim(ax.get_ylim())
    # ax2.yscale('log')
    locmaj = ax.yaxis.get_major_locator()
    numdecs = math.log10(locmaj()[-1]/locmaj()[0])
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator())
    ax2.yaxis.get_major_locator().set_params(numdecs=numdecs, numticks=numdecs)
    # print(locmaj())
    # print(ax2.yaxis.get_major_locator()())
    # locmin = ax.yaxis.get_minor_locator()
    subs = np.arange(1, 10)
    # locmin.set_params(subs=subs, numdecs=numdecs, numticks=len(subs)*numdecs)
    ax2.yaxis.set_minor_locator(matplotlib.ticker.LogLocator())
    ax2.yaxis.get_minor_locator().set_params(subs=subs, numdecs=numdecs,
                                             numticks=len(subs)*numdecs)
    # print(locmin())
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    # ax2.set(yticks=ax.get_yticks())
    # ax2.set_yticks(ax.get_yticks())
    # print(ax2.get_yticks())
    plt.grid(visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

def plot_compare(filename, savename, **kwargs):
    print(filename)
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    labels = collections.OrderedDict([
        ('residual_swept', 'Residual, swept'),
        ('residual_streamed', 'Residual, streamed'),
        ('residual', 'Residual'),
        ('norm', 'Mode $M-1$'),
        ('error_svd_m', 'Error $\\phi$, SVD'),
        ('error_svd_d', 'Error $\\psi$, SVD'),
        ('error_m', 'Error $\\phi$, PGD'),
        ('error_d', 'Error $\\psi$, PGD')])
    # for name in table.dtype.names:
    for name in labels.keys():
        if name not in table.dtype.names:
            continue
        kwargs_line = kwargs
        # if name != 'error_d':
        # if ('error' not in name 
        #         and 'norm' not in name
        #         and 'residual_streamed' not in name
        #         ):
        #     continue
        # if 'error' not in name:
        #     continue
        kwargs_line['label'] = labels.get(name, name.replace('_', '\\_'))
        kwargs_line['ls'] = '-'
        kwargs_line['marker'] = 'o'
        kwargs_line['alpha'] = 0.8
        # kwargs_line['markersize'] = 2.75
        # kwargs_line['markevery'] = 2
        if 'error' in name:
            kwargs_line['color'] = 'C0'
        if 'residual' in name:
            kwargs_line['color'] = 'C1'
            if 'streamed' in name:
                # kwargs_line['ls'] = '--'
                kwargs_line['marker'] = 'D'
                kwargs_line['alpha'] = 0.5
            if 'swept' in name:
                # kwargs_line['ls'] = ':'
                kwargs_line['marker'] = 's'
                kwargs_line['alpha'] = 0.5
        if 'norm' in name:
            kwargs_line['color'] = 'C2'
        if 'svd' in name:
            kwargs_line['color'] = 'C3'
        if '_m' in name:
            # kwargs_line['ls'] = '--'
            kwargs_line['marker'] = 'D'
            kwargs_line['alpha'] = 0.5
            # continue
        # if name != 'error_d' and name != 'error_svd_d':
        #     continue
        # if name[-1] == 'm':
        #     kwargs['color'] = plt.gca().lines[-1].get_color()
        #     kwargs['ls'] = '--'
        xdata = enrichments
        ydata = table[name] / table['error_d'][0]
        ydata = np.abs(ydata)
        if ydata[0] == 0:
            ydata = ydata[1:]
            xdata = xdata[:-1]
        # ydata /= ydata[0]
        plt.plot(xdata, ydata, **kwargs_line)
        if 'error' in name and 'svd' not in name:
            print(name, ' & '.join('%.2e' % y for y in ydata[10::10]))
    # plt.xticks(refinements)
    # plt.legend(loc='best')
    plt.yscale('log')
    # plt.title('L2 Convergence')
    # plt.ylabel('$L2$ Error')
    # plt.xlabel('Modes $M$')
    plt.tight_layout(pad=0.2)
    locmaj = plt.gca().yaxis.get_major_locator()
    locs = locmaj()
    numdecs = math.log10(locs[-1] / locs[0])
    locmaj.set_params(numdecs=numdecs, numticks=numdecs)
    locmin = plt.gca().yaxis.get_minor_locator()
    locmin.set_params(subs=np.arange(1, 10), numdecs=numdecs, numticks=9*numdecs)
    plt.savefig(savename)
    # plt.close()

def main(fuel, ext):
    name_base = 'GroupStructure_CathalauCompareTest{algorithm}_{fuel}_{param}'
    algorithms = ('Progressive', 'WithUpdate') #('Progressive', 'WithUpdate')
    # params = range(9)
    params = ['CASMO-'+str(num) for num in (70,)]
    params += ['XMAS-172', 'SHEM-361'] # 'CCFE-709', 'UKAEA-1102']
    nrows = len(params)
    ncols = len(algorithms)
    ij = 0
    for i, param in enumerate(params):
        for j, algorithm in enumerate(algorithms):
            # if param == 'SHEM-361' and algorithm == 'WithUpdate':
            #     break
            ij += 1
            if ij == 1:
                axij = plt.subplot(nrows, ncols, ij)
                axj0 = axij
                axij0 = axij
            elif j == 0:
                axij = plt.subplot(nrows, ncols, ij)
                axj0 = axij
            else:
                axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
            name = name_base.format(algorithm=algorithm, param=param, fuel=fuel)
            ls = '--' if algorithm == 'WithUpdate' else None
            ls = None
            label = param if algorithm == 'Progressive' else None
            # label = param
            # color = plt.gca().lines[-1].get_color() if algorithm == 'WithUpdate' \
            #         else None
            # color = 'C'+str(i)
            color = None
            plt.gca().set_prop_cycle(None)
            plot_compare(name+'.txt', name+'.'+ext, 
                         label=label, color=color, 
                         markevery=2, markersize=2.75)
            # plt.gca().yaxis.get_ticklocs(minor=True)
            # plt.gca().minorticks_on()
            plt.gca().tick_params(axis='y', which='both', left=True)
            if j > 0:
                plt.setp(axij.get_yticklabels(), visible=False)
            else:
                pass
            if j == ncols - 1:
                axij.yaxis.set_label_position('right')
                plt.ylabel(param)
            if i < nrows - 1:
                plt.setp(axij.get_xticklabels(), visible=False)
                plt.gca().xaxis.set_ticklabels([])
            else:
                pass
            if i == 0:
                fancy = {'Progressive': 'Progressive', 
                         'WithUpdate': 'With Update'}
                plt.title(fancy[algorithm])
            # plt.close()
            handles, desc = plt.gca().get_legend_handles_labels()
            if j == ncols - 1:
                pass
                right_yticks()
                plt.setp(axij.get_yticklabels(), visible=False)
    rect = [0.03, 0.025, 1, 0.925]
    if JCP:
        rect[-1] = 0.915
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                     rect=rect)
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Modes $M$', labelpad=20 if not JCP else 17.5)
    ax0.set_ylabel('Normalized $L^2$ Error', labelpad=32.5 if not JCP else 30)
    bbox_to_anchor = [0.49, 1.14]
    if JCP:
        bbox_to_anchor[1] = 1.165
    legend = ax0.legend(reversed(handles), reversed(desc), loc='upper center', 
                        ncol=math.ceil(len(handles)/2),
                        bbox_to_anchor=bbox_to_anchor)
    # plt.tight_layout(pad=0.02)
    plt.savefig('compare-{fuel}.pdf'.format(fuel=fuel))

if __name__ == '__main__':
    # python plot_compare.py uo2 pdf
    # python plot_comapre.py mox43 pdf
    plt.style.use('thesis.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 6.375))
    if sys.argv[-1] == 'jcp':
        sys.argv.pop()
        JCP = True
        plt.style.use('jcp.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 5.5))
        matplotlib.rc('legend', fontsize=10)
    main(*sys.argv[1:])
    # plot_compare(*sys.argv[1:])