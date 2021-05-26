import collections

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from plot_compare import right_yticks


def style(name):
    kwargs = {}
    kwargs['marker'] = 'o'
    kwargs['alpha'] = 0.8
    if 'error' in name:
        kwargs['color'] = 'C0'
        if '_m' in name:
            kwargs['marker'] = 'D'
            kwargs['alpha'] = 0.5
    if 'residual' in name:
        kwargs['color'] = 'C1'
        if 'streamed' in name:
            kwargs['marker'] = 'D'
            kwargs['alpha'] = 0.5
        elif 'swept' in name:
            kwargs['marker'] = 's'
            kwargs['alpha'] = 0.5
    if 'norm' in name:
        kwargs['color'] = 'C2'
    if 'svd' in name:
        kwargs['color'] = 'C3'
    return kwargs


def plot_compare_ss(filebase, savename, **kwargs):
    enrichments = [0, 5] + list(range(10, 51, 10))
    print(enrichments)
    labels = collections.OrderedDict([
        ('residual_swept', 'Residual, swept'),
        # ('residual_streamed', 'Residual, streamed'),
        # ('residual', 'Residual'),
        # ('norm', 'Mode $M-1$'),
        # ('error_svd_m', 'Error $\\phi$, SVD'),
        # ('error_svd_d', 'Error $\\psi$, SVD'),
        ('error_m', 'Error $\\phi$, PGD'),
        ('error_d', 'Error $\\psi$, PGD')])
    dtype = np.dtype([(name, 'f') for name in labels.keys()])
    # fuels = ['uo2', 'mox43']
    guesses = ['Pgd', 'Svd']
    structures = ['CASMO-70', 'XMAS-172', 'SHEM-361']
    pos = 0
    num_rows = len(structures)
    num_cols = len(guesses)
    for row, structure in enumerate(structures):
        for col, guess in enumerate(guesses):
            pos += 1
            table_enrs = np.zeros(len(enrichments), dtype)
            plt.subplot(num_rows, num_cols, pos)
            for i, enrichment in enumerate(enrichments[1:]):
                filename = filebase.format(guess=guess, structure=structure, 
                                           num_modes=enrichment)
                table_enr = np.genfromtxt(filename, names=True)
                for name in labels.keys():
                    table_enrs[name][i+1] = table_enr[name][-1]
                    if i == 0:
                        table_enrs[name][0] = table_enr[name][0]
            initial = table_enr['error_d'][0]
            for name in labels.keys():
                table_enrs[name] /= initial
                plt.plot(enrichments, table_enrs[name][:], label=labels[name],
                         **style(name))
            for name in ['error_svd_m', 'error_svd_d']:
                table_enr[name] /= initial
                plt.plot(table_enr[name], markevery=2, **style(name))
            plt.yscale('log')
            # plt.savefig(f'{savename}_{fuel}_{structure}.pdf')
            if row == 0:
                plt.title(guess)
            if row < num_rows - 1:
                plt.setp(plt.gca().get_xticklabels(), visible=False)
                plt.gca().xaxis.set_ticklabels([])
            if col > 0:
                plt.setp(plt.gca().get_yticklabels(), visible=False)
                plt.gca().yaxis.set_ticklabels([])
            if col == num_cols - 1:
                plt.gca().yaxis.set_label_position('right')
                plt.ylabel(structure)
    plt.tight_layout(pad=0.1)
    plt.savefig(savename+'.pdf')
    plt.close()


def main():
    plt.style.use('jcp.mplstyle')
    matplotlib.rc('figure', figsize=(6.5, 5.5))
    matplotlib.rc('legend', fontsize=10)
    # filebase = 'GroupStructure_CathalauCompareTestWithUpdate' \
    #            '_{fuel}_{structure}_M{num_modes}-f.txt'
    filebase = 'GroupStructureModes_CathalauCompareSubspaceTest{guess}Energy' \
                '_%s_{structure}_M{num_modes}.txt'
    fuels = ['uo2', 'mox43']
    for fuel in fuels:
        plot_compare_ss(filebase % fuel, 'error_pgd_svd_block_gs_'+fuel)


if __name__ == '__main__':
    main()

    