import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt

import openmc.mgxs

import plot_mgxs

JCP = True

def plot_all(loc, fuels):
    # fuels are columns
    # group structures are rows
    base = loc + '/{fuel}-v2/mgxs-uncorrected-{group_structure}.h5'
    group_structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    nrows = len(group_structures)
    ncols = len(fuels)
    ij = 0
    for i, group_structure in enumerate(group_structures):
        for j, fuel in enumerate(fuels):
            ij += 1
            if ij == 1:
                ax = plt.subplot(nrows, ncols, ij)
                axij = ax
            else:
                axij = plt.subplot(nrows, ncols, ij, sharex=ax, sharey=ax)
            name = base.format(fuel=fuel, group_structure=group_structure)
            plot_mgxs.plot_mgxs(name, ['water', fuel, 'zr', 'al'][::-1])
            lines = plt.gca().get_lines()
            for c, line in enumerate(lines):
                ci = len(lines) - 1 - c
                line.set_color('C'+str(ci))
            # lines[0].set_color('C2')  # Al
            # lines[2].set_color('C0')  # Water
            if j > 0:
                plt.setp(axij.get_yticklabels(), visible=False)
            else:
                pass
            if j == ncols - 1:
                axij.yaxis.set_label_position('right')
                plt.ylabel(group_structure)
            if i < nrows - 1:
                plt.setp(axij.get_xticklabels(), visible=False)
            else:
                pass
            if i == 0:
                fancy = {'uo2':r'UO\textsubscript{2}', 
                         'mox43': r'4.3\% MOX'}
                plt.title(fancy[fuel])
    lines = []
    materials = ['Fuel', 'Water', 'Zr', 'Al']
    colors = ['C' + str(i) for i in (1, 0, 2, 3)]
    for i, material in enumerate(materials):
        line = matplotlib.lines.Line2D([], [], color=colors[i])
        lines.append(line)
    # plt.figlegend(lines, materials, loc='upper center',
    #               ncol=len(materials))
    # plt.yscale('linear')
    plt.tight_layout(pad=0.2, rect=(0.03, 0.04, 1, 0.92 if not JCP else 0.925))
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Energy [eV]', labelpad=20 if not JCP else 18)
    ax0.set_ylabel(r'Total Cross Section $\Sigma_t$ [cm\textsuperscript{-1}]', 
                   labelpad=25 if not JCP else 23)
    legend = ax0.legend(lines, materials, loc='upper center', ncol=len(materials),
                        bbox_to_anchor=(0.5, 1.17) if not JCP else (0.5, 1.175))

if __name__ == '__main__':
    # python plot_mgxs_all.py mgxs-all.pdf ../openmc-c5g7 uo2 mox43
    import mpl_rc
    normal = 12
    small = 10.95
    footnotesize = 10
    figsize = (6.5, 4.5)
    mpl_rc.set_rc(small, figsize)
    # matplotlib.rc('legend', fontsize=footnotesize, title_fontsize=footnotesize)
    matplotlib.rc('axes', titlesize=small)
    if JCP:
        plt.style.use('./examples/cathalau-uo2/jcp.mplstyle')
        matplotlib.rc('figure', figsize=(6.5, 4.125))
    print(sys.argv)
    # matplotlib.rc()
    plot_all(sys.argv[2], sys.argv[3:])
    plt.savefig(sys.argv[1])
  