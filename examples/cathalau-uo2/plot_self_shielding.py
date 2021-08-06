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
    cross_section = cross_section[g_min-1:g_max]
    widths = np.log(group_structure[:-1]/group_structure[1:])
    # widths = group_structure[:-1] - group_structure[1:]
    mids = (group_structure[1:] + group_structure[:-1]) / 2
    xs_coarses = collections.OrderedDict()
    labels = collections.OrderedDict()
    if study == 'angular':
        labels['spectra_iso'] = 'Isotropic'
        labels['spectra_in'] = 'In'
        labels['spectra_out'] = 'Out'
    elif study == 'spatial':
        labels['spectra_pin'] = 'Pin'
        labels['spectra_iso'] = 'Edge'
        labels['spectra_center'] = 'Center'
    ax0 = plt.gca()
    ax1 = plt.twinx()
    for name in labels.keys():
        if 'spectra' not in name:
            continue
        ydata = np.array(table[name])
        if 'pin' in name:
            area_defect = lambda n : n/(2*math.pi) * math.sin(2*math.pi/n)
            ydata *= area_defect(64)
        # print(len(group_structure), len(ydata))
        denom = sum(ydata)
        collision = sum(cross_section*ydata)
        # collision = sum(table['cross_section']*ydata)
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
        plt.sca(ax1)
        xs_coarse = collision / denom
        plt.axhline(xs_coarse, ls='--', color=color, alpha=0.4)
        xs_coarses[name] = xs_coarse
        plt.sca(ax0)
        plt.step(group_structure, np.append(ydata[0], ydata), color=color,
                 ls=ls, where='pre', 
                 label=labels.get(name, name.replace('_', '\\_')), alpha=alpha)
    names = list(labels.keys())
    diffs = []
    for name in names[1:]:
        diff = (xs_coarses[name] - xs_coarses[names[0]]) / xs_coarses[names[0]]
        print(name, diff*100)
        diffs.append(f'{diff*100:+.1f}\\%')
    # ax1.text(0.6, 0.5, ', '.join(r'\textcolor{green} %s ' % d for d in diffs), 
    #          transform=ax1.transAxes)
    # ax0.get_xaxis().get_major_formatter().set_powerlimits([1, 6])
    # ax1.get_xaxis().get_major_formatter().set_powerlimits([1, 6])
    # ax1.get_xaxis().get_major_formatter().set_scientific(False)
    print(ax0.get_xaxis().get_major_formatter().__class__)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xscale('log')
    plt.yscale('log')
    bbox = (0.075, 0.5)
    legend = plt.legend(loc='center left', bbox_to_anchor=bbox)
    legend.remove()
    plt.grid(visible=False, )
    plt.tight_layout(pad=0.2)
    plt.gca().set_zorder(100)
    # plt.twinx()
    plt.sca(ax1)
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
    # xy_axes = (0.65, 0.5)
    # xy_data = ax1.transAxes.transform(xy_axes)
    # transform = ax1.transAxes
    # print(xy_data)
    # x0 = 0.65
    # x = x0
    # nice_diffs = []
    # for diff in diffs:
    #     nice_diffs.append(diff)
    #     nice_diffs.append(',  ')
    # nice_diffs.pop()
    # for diff, color in zip(nice_diffs, ['tab:green', 'k', 'tab:red', 'k']):
    #     text = ax0.text(x, 0.5, diff, color=color, transform=transform)
    #     text.draw(ax0.figure.canvas.get_renderer())
    #     extent = text.get_window_extent()
    #     x = transform.inverted().transform(extent)[1][0] + 0.0125
    #     print(transform.inverted().transform(extent))
        # print(extent.width)
        # transform = matplotlib.transforms.offset_copy(
        #     text.get_transform(), x=extent.width, units='dots')
    # ax0.add_patch(matplotlib.patches.FancyBboxPatch(
    #     (extent.xmin, extent.ymin), extent.width, extent.height, 
    #     boxstyle="square,pad=0.", fc='red', transform=ax0.transAxes.inverted(),
    #     zorder=10))
    diffs[0] += ','
    legend = ax0.legend([matplotlib.lines.Line2D([], [], visible=False)]*2, diffs, 
                        loc='center right', ncol=2, columnspacing=0.5, 
                        handlelength=0, handletextpad=0, 
                        bbox_to_anchor=(0.99, 0.5)
                        )
    for text, color in zip(legend.get_texts(), ['tab:green', 'tab:red']):
        text.set_color(color)
    # colors = ['tab:green', 'tab:red']
    # for i in reversed(list(range(len(diffs)))):
    #     if i:
    #         ax1.text(0.7, 0.5, ', '.join(diffs[:i])+',', color='k', transform=ax1.transAxes)
    #     text = ', '.join(diffs[:i+1])
    #     ax1.text(0.7, 0.5, text, color=colors[i], transform=ax1.transAxes)



if __name__ == '__main__':
    # python plot_self_shielding.py SelfShieldingTestDiagonal foo
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
    matplotlib.rc('font', size=size)
    # matplotlib.rc('text', usetex=False)
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
                     rect=(0.04, 0.04, 0.95, 1.0))
    # plt.gca().get_xaxis().get_major_formatter().set_min_exponent(1)
    plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Energy [eV]', labelpad=12)
    ax0.set_ylabel(r'Neutron Flux [1/lethargy]', labelpad=25)
    # ax0.set_ylabel(r'Neutron Flux [n/cm\textsuperscript{2}-s]', labelpad=25)
    # ax0.set_ylabel('Neutron Flux per Unit Lethargy', labelpad=25)
    axt = plt.twinx()
    axt.set_frame_on(False)
    axt.set_yticks([])
    plt.ylabel(r'Total Cross Section $\Sigma_t$ [cm\textsuperscript{-1}]', color='C4', 
               labelpad=20)
    plt.savefig(f'{name}-{study}.pdf')