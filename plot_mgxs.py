import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import openmc.mgxs

def plot_mgxs(filename, names=None):
    file = h5py.File(filename, 'r')
    num_groups = file.attrs['energy_groups']
    if 'group structure' in file.attrs:
        group_structure = file.attrs['group structure']
        xdata = group_structure[::-1]#[:-1]
        plt.xscale('log')
    else:
        xdata = range(num_groups)
    if names is None:
        names = file.keys()
    for name in names:
        if name == 'void':
            continue
        material = file[name]
        ydata = np.array(material['294K']['total'])
        plt.step(xdata, np.append(ydata[0], ydata), where='pre', label=name)
        # if material.attrs['fissionable']:
        #     plt.step(xdata, material['294K']['chi'], where='mid', 
        #              label=name+' $\\chi$')
    # groups_coarse = openmc.mgxs.GROUP_STRUCTURES['CASMO-70']
    # for g_rev, bound in enumerate(groups_coarse):
    #     g = (len(groups_coarse)-1) - g_rev
    #     # if bound > 3.5 and bound < 400:
    #     plt.axvline(bound, ls='--', color='black')
    # plt.xlim(plt.xlim()[0], 400)
    plt.yscale('log')

def set_size(w, h, ax):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

if __name__ == '__main__':
    plt.style.use('./examples/cathalau-uo2/jcp.mplstyle')
    width = 1.25984*2.25/2.54 * 1.2
    height = width * 0.7
    # matplotlib.rc('figure', figsize=(width, 1))
    size = 6
    matplotlib.rc('lines', lw=0.8)
    majorwidth = 0.6
    minorwidth = 0.4
    majorpad = 0.65
    matplotlib.rc('xtick', labelsize=size)
    matplotlib.rc('xtick.major', width=majorwidth, pad=majorpad)
    matplotlib.rc('xtick.minor', width=minorwidth)
    matplotlib.rc('ytick', labelsize=size)
    matplotlib.rc('ytick.major', width=majorwidth, pad=majorpad)
    matplotlib.rc('ytick.minor', width=minorwidth)
    matplotlib.rc('grid', linewidth=majorwidth)
    matplotlib.rc('axes', linewidth=majorwidth, labelsize=size, titlesize=size,
                          labelpad=majorpad, titlepad=1.2)
    plot_mgxs(sys.argv[1], ['uo2'])
    plt.ylabel(r'Cross-Section [cm\textsuperscript{-1}]\hspace{-5cm}', 
               labelpad=majorpad)
    plt.xlabel('Energy [eV]')
    # majorformatter_old = plt.gca().get_yaxis().get_major_formatter()
    # majorformmater_new = 
    ticker = matplotlib.ticker.LogFormatterSciNotation(labelOnlyBase=True)
    # ticker_major = matplotlib.ticker.LogFormatterExponent(labelOnlyBase=True)
    plt.gca().get_yaxis().set_minor_formatter(ticker)
    # plt.gca().get_yaxis().set_label_coords(-0.2, 0.45)
    # plt.gca().get_yaxis().set_major_formatter(ticker_major)
    # plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    set_size(width, height, plt.gca())
    plt.tight_layout(pad=0.075)
    # plt.legend(loc='best')
    plt.savefig(sys.argv[2])
  