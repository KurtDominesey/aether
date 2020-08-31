import math
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, '../cathalau-uo2')
import plot_compare

def main(ext):
    name_base = 'C5G7CompareTest{algorithm}'
    algorithms = ('Progressive', 'WithUpdate')
    nrows = 1
    ncols = len(algorithms)
    i = 0
    ij = 0
    for j, algorithm in enumerate(algorithms):
        # if param == 'SHEM-361' and algorithm == 'WithUpdate':
        #     break
        ij += 1
        if ij == 1:
            axij = plt.subplot(nrows, ncols, ij)
            axj0 = axij
            axij0 = axij
        elif j == 0:
            axij = plt.subplot(nrows, ncols, ij)
            axj0 = axij
        else:
            axij = plt.subplot(nrows, ncols, ij, sharey=axj0)
        name = name_base.format(algorithm=algorithm)
        ls = '--' if algorithm == 'WithUpdate' else None
        ls = None
        # label = param
        # color = plt.gca().lines[-1].get_color() if algorithm == 'WithUpdate' \
        #         else None
        # color = 'C'+str(i)
        color = None
        plt.gca().set_prop_cycle(None)
        plot_compare.plot_compare(name+'.txt', name+'.'+ext,
                                  markersize=2.75)
        plt.ylim(9.857432187773571e-08, 2.155908353722149)  #!!!
        # plt.gca().yaxis.get_ticklocs(minor=True)
        # plt.gca().minorticks_on()
        plt.gca().tick_params(axis='y', which='both', left=True)
        if j > 0:
            plt.setp(axij.get_yticklabels(), visible=False)
        else:
            pass
        if j == ncols - 1:
            axij.yaxis.set_label_position('right')
        if i < nrows - 1:
            plt.setp(axij.get_xticklabels(), visible=False)
            plt.gca().xaxis.set_ticklabels([])
        else:
            pass
        if i == 0:
            fancy = {'Progressive': 'Progressive', 
                      'WithUpdate': 'With Update'}
            plt.title(fancy[algorithm])
        # plt.close()
        handles, desc = plt.gca().get_legend_handles_labels()
        if j == ncols - 1:
            pass
            plot_compare.right_yticks()
            plt.setp(axij.get_yticklabels(), visible=False)
    plt.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5, 
                     rect=(0.028, 0.05, 1, 0.825)
                     )
    ax0 = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_xlabel('Modes $M$', labelpad=20)
    ax0.set_ylabel('Normalized $L^2$ Error', labelpad=32.5)
    legend = ax0.legend(reversed(handles), reversed(desc), loc='upper center', 
                        ncol=math.ceil(len(handles)/2),
                        bbox_to_anchor=(0.48, 1.45))
    # plt.tight_layout(pad=0.02)
    plt.savefig('compare.pdf')

if __name__ == '__main__':
    # python plot_compare_cathalau.py pdf
    # plt.style.use('../cathalau-uo2/thesis.mplstyle')
    plt.style.use('../cathalau-uo2/thesis.mplstyle')
    # matplotlib.rc('figure', figsize=(6.5, 2+7/8))
    matplotlib.rc('figure', figsize=(6.5, 2.75))
    main(*sys.argv[1:])
    # plot_compare(*sys.argv[1:])