import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_compare(filename, savename, **kwargs):
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    for name in table.dtype.names:
        # if name != 'error_d':
        if 'j2' not in name:
            continue
        # plt.title('uo2')
        # if name[-1] == 'm':
        #     kwargs['color'] = plt.gca().lines[-1].get_color()
        #     kwargs['ls'] = '--'
        marker = '.' #'o'
        xdata = enrichments
        ydata = table[name] #/ table['error_d'][0]
        ydata = np.abs(ydata)
        if ydata[0] == 0:
            ydata = ydata[1:]
            xdata = xdata[:-1]
        # ydata /= ydata[0]
        if 'inf' in name:
            marker = None
            kwargs['ls'] = '--'
        elif 'svd' in name:
            kwargs['ls'] = ':'
        else:
            kwargs['ls'] = '-'
        g_name = ''
        for char in name.split('g')[-1]:
            if not char.isdigit():
                break
            g_name += char
        g = int(g_name)
        if g < 21 or g > 27:
            continue
        if 'svd' in name:
            continue
        color = 'C' + str(g % 10)
        kwargs['color'] = color
        label = None if ('inf' in name or 'svd' in name) else str(g)
        plt.plot(xdata, ydata, label=label, marker=marker, **kwargs)
        # plt.plot(np.extract(table[name] > 0, xdata),
        #          np.extract(table[name] > 0, ydata), ls=None,
        #          marker='+', markersize=10)
    # plt.xticks(refinements)
    plt.legend(loc='best', title='Group')
    # plt.ylim(1e-5, plt.ylim()[1])
    plt.yscale('log')
    plt.xlabel('Modes')
    plt.ylabel('Relative Error')
    plt.savefig(savename)
    # plt.close()

if __name__ == '__main__':
    # main(sys.argv[1])
    plot_compare(*sys.argv[1:])