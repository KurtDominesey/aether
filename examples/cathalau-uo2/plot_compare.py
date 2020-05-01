import sys

import numpy as np
import matplotlib.pyplot as plt

def plot_compare(filename, savename, **kwargs):
    table = np.genfromtxt(filename, names=True)
    enrichments = list(range(table.shape[0]))
    for name in table.dtype.names:
        # if name != 'error_d':
        # if ('error' not in name 
        #         and 'norm' not in name
        #         and 'residual_streamed' not in name
        #         ):
        #     continue
        if 'error' not in name:
            continue
        # if name != 'error_d' and name != 'error_svd_d':
        #     continue
        # if name[-1] == 'm':
        #     kwargs['color'] = plt.gca().lines[-1].get_color()
        #     kwargs['ls'] = '--'
        kwargs['label'] = name
        marker = '.' #'o'
        xdata = enrichments
        ydata = table[name] / table['error_d'][0]
        ydata = np.abs(ydata)
        if ydata[0] == 0:
            ydata = ydata[1:]
            xdata = xdata[:-1]
        # ydata /= ydata[0]
        plt.plot(xdata, ydata, marker=marker, **kwargs)
    # plt.xticks(refinements)
    plt.legend(loc='best')
    plt.yscale('log')
    plt.title('L2 Convergence')
    plt.ylabel('L2 Error')
    plt.xlabel('Modes')
    plt.savefig(savename)
    # plt.close()

def main(ext):
    name_base = 'GroupStructure_CathalauCompareTest{algorithm}_{param}'
    algorithms = ('Progressive', 'WithUpdate') #('Progressive', 'WithUpdate')
    # params = range(9)
    params = ['CASMO-'+str(num) for num in (8, 16, 25, 40, 70)]
    params += ['XMAS-172', 'SHEM-361'] # 'CCFE-709', 'UKAEA-1102']
    for i, param in enumerate(params):
        for algorithm in algorithms:
            name = name_base.format(algorithm=algorithm, param=param)
            ls = '--' if algorithm == 'WithUpdate' else None
            # ls = None
            label = param if algorithm == 'Progressive' else None
            # label = param
            # color = plt.gca().lines[-1].get_color() if algorithm == 'WithUpdate' \
            #         else None
            # color = 'C'+str(i)
            color = None
            plt.gca().set_prop_cycle(None)
            plot_compare(name+'.txt', name+'.'+ext, 
                         ls=ls, label=label, color=color)
        plt.close()

if __name__ == '__main__':
    main(sys.argv[1])
    # plot_compare(*sys.argv[1:])