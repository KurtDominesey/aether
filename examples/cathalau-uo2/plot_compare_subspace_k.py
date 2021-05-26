import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import h5py as h5

from plot_compare import right_yticks


ALPHA = 0.8
colors = {'CASMO-70': 'C0', 'XMAS-172': 'C1', 'SHEM-361': 'C2'}
markers = {'Pgd': 'o', 'Svd': 's'}
# styles = {'uo2': '-', 'mox43': '-'}
styles = {'Energy': '-', 'Both': '--'}


def style(fuel, structure, guess, dim):
    kwargs = {}
    kwargs['marker'] = markers[guess]
    kwargs['alpha'] = 0.5 if dim == 'both' else ALPHA
    kwargs['color'] = colors[structure]
    kwargs['ls'] = styles[dim]
    return kwargs


def make_legends(fuels, structures, guesses, dims):
    legends = []
    lines_structure = []
    zero = ((0,), (0,))
    for structure in structures:
        lines_structure.append(Line2D(*zero, color=colors[structure], 
                                      alpha=ALPHA))
    legend_structure = plt.legend(lines_structure, structures, 
                                  bbox_to_anchor=(1, 1))
    lines_guess = []
    for guess in guesses:
        lines_guess.append(Line2D(*zero, marker=markers[guess], color='k', 
                                  alpha=ALPHA))
    legend_guess = plt.legend(lines_guess, guesses, bbox_to_anchor=(0.75, 1))
    lines_dim = []
    for dim in dims:
        lines_dim.append(Line2D(*zero, ls=styles[dim], color='k', 
                                 alpha=ALPHA))
    legend_dim = plt.legend(lines_dim, dims, bbox_to_anchor=(0.575, 1))
    return [legend_structure, legend_guess, legend_dim]


def plot_compare_subspace_k(savename):
    enrichments = [5] + list(range(10, 51, 10))
    print(enrichments)
    fuels = ['uo2', 'mox43']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    guesses = ['Pgd', 'Svd']
    dims = ['Energy'] #, 'Both']
    for fuel in fuels:
        for structure in structures:
            # if fuel == 'uo2' and structure != 'CASMO-70':
            #     continue
            # if fuel == 'mox43' and structure != 'XMAS-172':
            #     continue
            filename_full = f'{fuel}_{structure}_k_full.h5'
            file_full = h5.File(filename_full, 'r')
            k_full = file_full.attrs['k_eigenvalue']
            filebase = 'GroupStructureModes_CathalauCompareSubspaceTest'
            for guess in guesses:
                for dim in dims:
                    errors = np.zeros(len(enrichments))
                    for i, enr in enumerate(enrichments):
                        filename = filebase \
                            + f'{guess}{dim}_{fuel}_{structure}_pgd_s_M{enr}.h5'
                        file_enr = h5.File(filename, 'r')
                        k_enr = file_enr.attrs['k_eigenvalue']
                        errors[i] = abs(k_enr - k_full)
                    errors *= 10**5
                    plt.plot(enrichments, errors, markevery=1,
                             **style(fuel, structure, guess, dim))
        plt.yscale('log')
        # plt.axhline(5, color='blue', ls='')
        plt.ylabel('Eigenvalue Error [pcm]')
        plt.xlabel('Modes, $M$')
        plt.gca().grid(True, 'both', 'y')
        for legend in make_legends(fuels, structures, guesses, dims)[:-1]:
            plt.gca().add_artist(legend)
        plt.tight_layout(pad=0.5)
        plt.savefig(savename.format(fuel=fuel))
        plt.close()
                        

if __name__ == '__main__':
    plt.style.use('jcp.mplstyle')
    # matplotlib.rc('figure', figsize=(6.5, 5.5))
    mpl.rc('legend', fontsize=10)
    plot_compare_subspace_k('pgd_svd_block_gs_k_{fuel}.pdf')
    