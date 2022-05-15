import sys
import itertools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_conv(filename, style_cols, is_timed=False, **kwargs):
    table = np.genfromtxt(filename, delimiter=',', names=True)
    norm = table['err_psi'][0]
    if is_timed:
        xdata = table['runtime']
        for i in range(1, xdata.size):
            xdata[i] += xdata[i-1]
        xdata /= 60
    else:
        xdata = list(range(table.shape[0]))
    # for i, name in enumerate(table.dtype.names):
    for (ls, col) in style_cols:
        plt.plot(xdata, table[col]/norm, ls=ls, **kwargs)


def draw_line(**kwargs):
    if 'c' not in kwargs and 'color' not in kwargs:
        return Line2D([], [], color='black', **kwargs)
    else:
        return Line2D([], [], **kwargs)


def draw_legend(mark_dim_pol, color_dim_mg, style_cols):
    fmt_all = dict(bbox_to_anchor=[0, 1.075, 1, 0.1], handlelength=1.0, 
                   handletextpad=0.4, columnspacing=1.0)
    lines_pol = []
    for (marker, dim_pol) in mark_dim_pol:
        lines_pol.append(draw_line(marker=marker, label=f'{dim_pol}D'))
    legend_pol = plt.legend(handles=lines_pol, title='Polar', loc='upper left',
                            ncol=2, **fmt_all)
    ax = plt.gca()
    ax.add_artist(legend_pol)
    lines_flux = []
    for (ls, col) in style_cols:
        label = 'Angular, $\\psi$' if 'psi' in col else 'Scalar, $\\phi$'
        lines_flux.append(draw_line(ls=ls, label=label))
    legend_flux = plt.legend(handles=lines_flux, title='Flux', 
                             loc='upper right', ncol=2, **fmt_all)
    ax.add_artist(legend_flux)
    fmt_all['bbox_to_anchor'][2] -= 0.15
    lines_mg = []
    for (color, dim_mg) in color_dim_mg:
        label = f'{dim_mg}D' if dim_mg in (1, 2) else 'Both'
        lines_mg.append(draw_line(c=color, label=label))
    plt.legend(handles=lines_mg, title='Multigroup', ncol=3, loc='upper center',
               **fmt_all)


def tick_every_decade(ax):
    locmaj = mpl.ticker.LogLocator(base=10, numticks=np.inf)
    ax.yaxis.set_major_locator(locmaj)
    locmin = mpl.ticker.LogLocator(
        base=10, subs=[i*0.1 for i in range(10)], numticks=np.inf)
    ax.yaxis.set_minor_locator(locmin)


if __name__ == '__main__':
    plt.style.use('../../../aether.mplstyle')
    mpl.rc('legend', fontsize=10)
    mpl.rc('figure', figsize=(6.5, 4.125))
    model = sys.argv[1]
    base = 'out/ParamsTest2D1D{}%s' % model
    if model == 'Pin':
        configs = ('UO2', 'MOX43')
        config_names = ('UO\\textsubscript{2}', '4.3\\% MOX')
    elif model == 'Lwr':
        configs = ('RodY', 'RodN')
        config_names = ('Rodded', 'Unrodded')
    else:
        raise ValueError()
    either = (True, False)
    dims_pol = (1, 2)
    dims_mg = (1, 2, 'B')
    cols = ('err_psi', 'err_phi')
    markers = ('o', 's')
    colors = tuple(f'C{i}' for i in range(3))
    styles = ('-', '--')
    mark_dim_pol = list(zip(markers, dims_pol))
    color_dim_mg = list(zip(colors, dims_mg))
    style_cols = list(zip(styles, cols))
    fmts = list(itertools.product(markers, colors))
    cases = list(itertools.product(dims_pol, dims_mg))
    base_pgd = base.format('Pgd')
    base_svd = base.format('Svd')
    fmt_all = dict(markevery=2, markersize=2.75)
    fig, axes = plt.subplots(2, 2, sharex='col', clip_on=False)
    for i, config in enumerate(configs):
        for j, is_timed in enumerate(reversed(either)):
            ax = axes[i][j]
            plt.sca(ax)
            for ((dim_pol, dim_mg), (marker, color)) in zip(cases, fmts):
                suffix = f'{config}Pol{dim_pol}Mg{dim_mg}Gal.csv'
                try:
                    plot_conv(base_pgd+suffix, style_cols, is_timed, 
                              marker=marker, c=color, alpha=0.8, **fmt_all)
                    if is_timed:
                        continue
                    plot_conv(base_svd+suffix, style_cols, is_timed, 
                            marker=marker, c=color, alpha=0.2, **fmt_all)
                except IOError:
                    pass
            plt.yscale('log')
            tick_every_decade(ax)
            ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterExponent())
            if j:
                ax.yaxis.set_label_position('right')
                plt.ylabel(config_names[i])
            if i:
                plt.xlabel('Wall time [min]' if is_timed else 'Modes, $M$')
    plt.gcf().add_subplot(111, frame_on=False, alpha=0, xticks=[], yticks=[])
    draw_legend(mark_dim_pol, color_dim_mg, style_cols)
    plt.tight_layout(rect=(0.04, 0, 1, 1), pad=0.2)
    plt.ylabel('Log\\textsubscript{10} of Normalized $L^2$ Error', labelpad=22.5)
    plt.savefig(f'out/conv{model}3.pdf')
    plt.close()