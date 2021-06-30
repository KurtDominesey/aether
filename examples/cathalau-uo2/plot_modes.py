import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import openmc

STRUCTURE = 'CASMO-70'

def plot_modes(filepath_h5):
    file_h5 = h5.File(filepath_h5, 'r')
    num_modes = file_h5.attrs['num_modes']
    groups = openmc.mgxs.GROUP_STRUCTURES[STRUCTURE]
    widths = np.log(groups[1:]/groups[:-1])
    # widths = groups[1:] - groups[:-1]
    widths = widths[::-1]
    for m in range(num_modes):
        # mode = np.array(file_h5[f'modes_energy{m}'])
        for suffix in ['_adj']:
            # mode *= np.array(file_h5[f'modes_energy{suffix}{m}']) * -1
            mode = np.array(file_h5[f'modes_energy{suffix}{m}'])
            mode /= widths
            # mode /= np.linalg.norm(mode)
            if suffix == '':
                norm_sa = np.linalg.norm(file_h5[f'modes_spaceangle{m}'])
                # mode *= norm_sa
            plt.step(groups[::-1], np.append(mode[0], mode), where='pre',
                     label=f'{m}{suffix}')
    plt.legend(loc='best')
    # plt.yscale('symlog')
    plt.xscale('log')
    plt.savefig(f'modes_energy_adj_{STRUCTURE}.pdf')


if __name__ == '__main__':
    plot_modes(f'GroupStructureModes_CathalauCompareSubspaceTestPgdEnergy_mox43_{STRUCTURE}_pgd_s_M5.h5')