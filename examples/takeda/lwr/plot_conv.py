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
    lines = []
    for (marker, dim_pol) in mark_dim_pol:
        lines.append(draw_line(marker=marker, label=f'Polar {dim_pol}D'))
    for (color, dim_mg) in color_dim_mg:
        label = 'Multigroup ' + (f'{dim_mg}D' if dim_mg in (1, 2) else 'Both')
        lines.append(draw_line(c=color, label=label))
    for (ls, col) in style_cols:
        label = 'Angular, $\\psi$' if 'psi' in col else 'Scalar, $\\phi$'
        lines.append(draw_line(ls=ls, label=label))
    plt.legend(handles=lines)


if __name__ == '__main__':
    either = (True, False)
    either_yn = ('Y', 'N')
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
    base = 'out/ParamsTest2D1DFixedSourceRod'
    fig, axes = plt.subplots(2, 2, sharey='row', sharex='col')
    for i, is_rodded in enumerate(either_yn):
        for j, is_timed in enumerate(reversed(either)):
            ax = axes[i][j]
            plt.sca(ax)
            for ((dim_pol, dim_mg), (marker, color)) in zip(cases, fmts):
                suffix = f'{is_rodded}Pol{dim_pol}Mg{dim_mg}Gal.csv'
                plot_conv(base+suffix, style_cols, is_timed, marker=marker, 
                          c=color, markevery=2, markersize=2.75, alpha=0.8)
            plt.yscale('log')
            if j:
                ax.yaxis.set_label_position('right')
                plt.ylabel({'Y': 'R', 'N': 'Unr'}[is_rodded]+'odded')
            if i:
                plt.xlabel('Time [s]' if is_timed else 'Modes')
        # draw_legend(mark_dim_pol, color_dim_mg, style_cols)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'out/conv.pdf')
    plt.close()