import collections
import math
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import openmc.mgxs

def plot_self_shielding(filename, study='angular'):
    fuel = 'uo2'
    folder = '/mnt/c/Users/kurt/Documents/projects/openmc-c5g7/'
    mgxs = h5py.File(folder + fuel + '/mgxs-uncorrected-SHEM-361.h5', 'r')
    cross_section = mgxs[fuel]['294K']['total']
    table = np.genfromtxt(filename, names=True, delimiter=', ')
    names = list(table.dtype.names)
    group_structure = openmc.mgxs.GROUP_STRUCTURES['SHEM-361'][::-1]
    #casmo-sh-70 coarse group 27
    g_min = 224
    g_max = 276
    group_structure = group_structure[g_min-1:g_max+1]
    cross_section = cross_section[g_min:g_max+1]
    widths = np.log(group_structure[:-1]/group_structure[1:])
    # widths = group_structure[:-1] - group_structure[1:]
    mids = (group_structure[1:] + group_structure[:-1]) / 2
    labels = collections.OrderedDict()
    if study == 'angular':
        labels['spectra_iso'] = 'Isotropic'
        labels['spectra_in'] = 'In'
        labels['spectra_out'] = 'Out'
    elif study == 'spatial':
        labels['spectra_iso'] = 'Edge'
        labels['spectra_center'] = 'Center'
    for name in labels.keys():
        if 'spectra' not in name:
            continue
        ydata = np.array(table[name])
        print(len(group_structure), len(ydata))
        denom = sum(ydata)
        collision = sum(cross_section*ydata)
        ydata /= widths
        color = 'k'
        alpha = 0.7
        ls = '-'
        ydata /= sum(table['spectra_iso'])
        if study == 'angular':
            if 'in' in name:
                color = 'tab:green'
            elif 'out' in name:
                color = 'tab:red'
        elif study == 'spatial':
            if 'iso' in name:
                color = 'tab:green'
            elif 'center' in name:
                color = 'tab:red'
        # ydata *= cross_section
        # plt.axhline(collision/denom, ls=ls, color=color, alpha=0.4)
        plt.step(group_structure, np.append(ydata[0], ydata), color=color,
                 ls=ls, where='pre', 
                 label=labels.get(name, name.replace('_', '\\_')), alpha=alpha)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xscale('log')
    plt.yscale('log')
    bbox = (0.075, 0.4 if study == 'angular' else 0.45)
    legend = plt.legend(loc='center left', bbox_to_anchor=bbox)
    legend.remove()
    plt.grid(visible=False, )
    plt.tight_layout(pad=0.2)
    plt.gca().set_zorder(100)
    plt.twinx()
    plt.autoscale(enable=True, axis='x', tight=True)
    # cross_section = np.array(table['cross_section'])
    xs_color = 'C4'
    plt.step(group_structure, np.append(cross_section[0], cross_section), 
             where='pre', color=xs_color, alpha=0.4)
    plt.gca().spines['right'].set_color(xs_color)
    plt.tick_params(axis='y', which='both', labelcolor=xs_color, color=xs_color)
    plt.yscale('log')
    plt.tight_layout(pad=0.2)
    plt.gca().set_axisbelow(True)
    plt.gca().add_artist(legend)

if __name__ == '__main__':
    plt.style.use('jcp.mplstyle')
    width = 3.8
    height = 2.6
    matplotlib.rc('figure', figsize=(width, height))
    size = 8
    ticksize = 6
    matplotlib.rc('lines', lw=1.4)
    majorwidth = 0.6
    minorwidth = 0.4
    majorpad = 2
    matplotlib.rc('xtick', labelsize=ticksize)
    matplotlib.rc('xtick.major', width=majorwidth, pad=majorpad)
    matplotlib.rc('xtick.minor', width=minorwidth)
    matplotlib.rc('ytick', labelsize=ticksize)
    matplotlib.rc('ytick.major', width=majorwidth, pad=majorpad)
    matplotlib.rc('ytick.minor', width=minorwidth)
    matplotlib.rc('grid', linewidth=majorwidth)
    matplotlib.rc('axes', linewidth=majorwidth, labelsize=size, titlesize=size,
                          labelpad=majorpad, titlepad=1.2)
    matplotlib.rc('legend', fontsize=size)
    plt.xlabel('Energy [eV]')
    # run
    name = sys.argv[1]
    study = sys.argv[2]
    ax1 = plt.subplot(2, 1, 1)
    plot_self_shielding(name+'.csv', 'angular')
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2 = plt.subplot(2, 1, 2, sharey=ax1)
    plot_self_shielding(name+'.csv', 'spatial')
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                     rect=(0.04, 0.055, 0.95, 1.0))
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Energy [eV]', labelpad=15)
    ax0.set_ylabel('Flux [1/lethargy]', labelpad=25)
    axt = plt.twinx()
    axt.set_frame_on(False)
    axt.set_yticks([])
    plt.ylabel(r'Cross-Section [cm\textsuperscript{-1}]', color='C4', 
               labelpad=20)
    plt.savefig(f'{name}-{study}.pdf')