import sys
import copy
import math
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import h5py
import openmc.mgxs

def plot_spectrum(filename, savename, fuel, **kwargs):
    folder = '/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/'
    file = h5py.File(folder + fuel + '-v2/mgxs-uncorrected-SHEM-361.h5', 'r')
    materials = ['void', 'water', 'uo2']
    table = np.genfromtxt(filename, names=True)
    xdata = openmc.mgxs.GROUP_STRUCTURES['SHEM-361'][::-1]
    widths = openmc.mgxs.GROUP_STRUCTURES['SHEM-361']
    # widths = np.subtract(widths[1:], widths[:-1])
    widths = np.log(np.divide(widths[1:], widths[:-1]))
    widths = widths[::-1]
    print(widths)
    plt.xscale('log')
    # groups_coarse = openmc.mgxs.GROUP_STRUCTURES['CASMO-70'][::-1]
    groups_coarse = np.load(folder+'casmo-sh-70.npy')
    for g, bound in enumerate(groups_coarse):
        # if (bound > 3.5 and bound < 400):
        if 20 <= g < 27:
            width = groups_coarse[g+1] - bound
            mid = (bound * groups_coarse[g+1])**0.5
            # mid = groups_coarse[g+1] #bound #
            # mid = np.geomspace(bound, groups_coarse[g+1], num=3)[1]
            color = ['c', 'y'][g%2]
            plt.axvspan(groups_coarse[g+1], bound, alpha=0.15,
                        facecolor=color, edgecolor=None)
            # plt.axvline(bound, ls=':', color='black', linewidth=1)
            # plt.axvline(groups_coarse[g+1], ls=':', color='black', linewidth=1)
            # y = 5e-8 * 10**((bound-groups_coarse[20])/(groups_coarse[27]-groups_coarse[20]))
            # ratio = -math.log(mid - groups_coarse[27]) / math.log(groups_coarse[20]-groups_coarse[27])
            ratio = (g - 20) / 7
            scale = 0.25
            y = (0.42-scale/2) + scale * ratio
            y = 0.5
            # if g == 20 or g == 26:
                # plt.text(mid, y, r'{%s}' % (g+1), fontsize=10, ha='center', transform=plt.gca().get_xaxis_transform())
            # if g == 23:
                # plt.text(mid, y, r'{\ldots}', fontsize=10, ha='center', transform=plt.gca().get_xaxis_transform())
    fuels_fancy = {'uo2': r'UO\textsubscript{2}', 'mox43': r'4.3\% MOX'}
    fuel_fancy = fuels_fancy[fuel]
    names = list(table.dtype.names) #+ ['openmc']
    for name in names:
        kwargs_line = copy.deepcopy(kwargs)
        kwargs_line['alpha'] = 0.8
        if name != 'openmc':
            ydata = table[name]
        else:
            sp = h5py.File(folder + 'uo2-v2/statepoint.4000.h5', 'r')
            tally = sp['tallies']['tally 1057']['results']  # 1057, 1070
            ydata = tally[361:(361+361), 0, 0]  # fuel
            # ydata = tally[361:(361+361), 1, 0] / ydata
            # ydata = tally[:361, 0, 0]  # water
            print(tally)
            ydata = ydata[::-1]
            # ydata = np.array(file['uo2']['294K']['chi'])
            print(sum(file['uo2']['294K']['chi']))
            # kwargs_line['ls'] = '--'
            # kwargs_line['alpha'] = 0.5
        denom = sum(ydata)
        # denom = sum(ydata**2/widths)**0.5
        # ydata *= ydata
        ydata /= widths
        ydata /= denom #np.linalg.norm(ydata, 2)
        if 'j' in name:
            j = int(name.split('j')[1])
            # if j < 1 or j > 2:
            if j != 2 and j != 1:
                continue
            material = materials[j]
            # if material == 2:
            #     kwargs['ls'] = '--'
        # if name == 'j0' or name == 'j3':  # void
        #     continue
        elif name == 'inf':
            pass
        elif 'mode' in name:
            if 'u' in name:
                continue
            kwargs_line['ls'] = ':'
            # kwargs_line['alpha'] = 0.5
        # xdata = range(len(ydata))
        # ydata *= file['uo2']['294K']['total']
        labels = {'j1': 'Water', 'j2': fuel_fancy, 'inf': 'Homogenized',
                  'mode1p': r'$\mathcal{E}_{m=1}$'}
        label = labels.get(name, name)
        plt.step(xdata, np.append(ydata[0], ydata), where='pre', label=label, 
                 **kwargs_line)
    # plt.xlim(3, 500)
    # plt.xlim(groups_coarse[27], groups_coarse[20])
    # plt.ylim(1e-6, 1e-3)
    plt.yscale('log')
    plt.legend(loc='center right', bbox_to_anchor=(0.9, 0.5))
    plt.ylabel('Normalized Spectrum [1/lethargy]')
    plt.xlabel('Energy [eV]')
    plt.tight_layout(pad=0.2)
    ax = plt.gca()
    twinx = plt.gca().twinx()
    total = file[fuel]['294K']['total']
    xs_color = 'C4'
    twinx.step(xdata, np.append(total[0], total), where='pre', color=xs_color, 
               alpha=0.5, label=fuel_fancy+r' $\Sigma_t$')
    plt.yscale('log')
    plt.gca().spines['right'].set_color(xs_color)
    plt.ylabel(r'Total Cross-Section $\Sigma_t$ [cm\textsuperscript{-1}]', color=xs_color)
    plt.tick_params(axis='y', which='both', labelcolor=xs_color, color=xs_color)
    plt.grid(visible=False)
    # plt.xlim(plt.xlim()[0], 1)
    # plt.xlim(1e-2, 1)
    # lines, labels = ax.get_legend_handles_labels()
    # line, label = plt.gca().get_legend_handles_labels()
    # lines += line
    # labels += label
    # plt.legend(lines, labels, loc='center left')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.2)
    plt.savefig(savename)

if __name__ == '__main__':
    plt.style.use('jcp.mplstyle')
    # matplotlib.rc('legend', fontsize=10.95)
    # matplotlib.rc('figure', figsize=(6.5, 2.75))
    matplotlib.rc('figure', figsize=(6.5, 2+7/8))
    plot_spectrum(*sys.argv[1:])
    # python plot_spectrum.py Fuel_CathalauMgxsTestToCasmo70_uo2_spectrum.txt Fuel_CathalauMgxsTestToCasmo70_uo2_spectrum-thesis.pdf uo2
    # python plot_spectrum.py Fuel_CathalauMgxsTestToCasmo70_mox43_spectrum.txt Fuel_CathalauMgxsTestToCasmo70_mox43_spectrum-thesis.pdf mox43