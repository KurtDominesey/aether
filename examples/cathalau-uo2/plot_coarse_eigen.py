import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COARSE_ERRORS = {
    'uo2-cp': [2.8349415852e-05, 1.1303810906e-05, 6.7128896424e-07],
    'uo2-ip': [2.6872510661e-05, 9.2828026425e-06, 1.9443769204e-06],
    # 'uo2-cp': [2.8349414788e-05, 1.1303807790e-05, 6.7131348459e-07], #[2.834933e-05, 1.130357e-05, 6.747873e-07],
    # 'uo2-ip': [ 2.6872516703e-05, 9.2828191263e-06, 1.9444144078e-06], #[2.834933e-05, 1.130357e-05, 6.747873e-07],
    'mox43-cp': [2.7380849360e-05, 1.0986546449e-05, 1.1885643240e-06],
    'mox43-ip': [2.6722866149e-05, 8.5678492688e-06, 1.6855109087e-06]
    # 'mox43-cp': [2.7380848040e-05, 1.0986542717e-05, 1.1885623885e-06], #[2.7380907222e-05, 1.0986762769e-05, 1.1859501066e-06],
    # 'mox43-ip': [2.6722863787e-05, 8.5678407364e-06, 1.6855014258e-06] #[2.6722035540e-05, 8.5649046124e-06, 1.6755661632e-06]
}

COARSE_DK = {
    'uo2-cp': 1.2996167473e+02,
    'uo2-ip': 5.2167117071e+02,
    # 'uo2-cp': 1.2997302432e+02, #1.315941e+02,
    # 'uo2-ip': 5.2168256969e+02, #1.315941e+02,
    'mox43-cp': 1.2300689344e+02,
    'mox43-ip': 4.3047816335e+02
    # 'mox43-cp': 1.2300433233e+02, #1.2095912371e+02,
    # 'mox43-ip': 4.3047477151e+02 #4.2800804587e+02
}


def share_ylim(ax1, ax2):
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    ymin = min(ymin1, ymin2)
    ymax = max(ymax1, ymax2)
    ymax = min(5, ymax)
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

def tick_every_decade(ax):
    locmaj = mpl.ticker.LogLocator(base=10, numticks=np.inf)
    locmin = mpl.ticker.LogLocator(base=10, numticks=np.inf,
                                   subs=[i*0.1 for i in range(10)])
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)


def Line(**kwargs):
    return mpl.lines.Line2D([], [], **kwargs)


def draw_axhlines_l2(ax, errors, norm, **kwargs):
    # errors = [2.834933e-05, 1.130357e-05, 6.747873e-07]
    for i, error in enumerate(errors):
        ax.axhline(error/norm, color=f'C{i}', **kwargs)
    return error


def draw_axhline_dk(ax, error, **kwargs):
    # error = 1.315941e+02 / 1e5
    ax.axhline(error/1e5, color='C4', **kwargs)


def plot_modal(filename, axl, axr, **kwargs):
    table = np.genfromtxt(filename, names=True)
    norm = table['l2_error_d'][0]
    for i, name in enumerate(table.dtype.names):
        if name == 'error_k':
            continue
        ydata = np.abs(table[name][:])
        ydata /= norm
        axl.plot(ydata, color=f'C{i}', **kwargs)
    axl.set_yscale('log')
    # plot 'error_k' on secondary axis
    ydata = np.abs(table['error_k'][:]) / 1e5
    col_k = 'C4'
    axr.plot(ydata, color=col_k, **kwargs)
    axr.grid(False)
    axr.set_yscale('log')
    axr.spines['right'].set_color(col_k)
    axr.tick_params(axis='y', which='both', labelcolor=col_k, color=col_k)
    return norm


def add_frame_labels(**kwargs):
    plt.tight_layout(pad=0.2, rect=(0, 0.05, 1, 0.82))
    ax = plt.gcf().add_subplot(1, 1, 1, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Modes $M$', labelpad=20)
    # add legend
    lines = [Line(color='C0', label='Error $\\psi$', **kwargs),
             Line(color='C1', label='Error $\\phi$', **kwargs),
             Line(color='C2', label='Error $q$', **kwargs),
             Line(color='C4', label='Error $k$', **kwargs),
             Line(color='k', ls='--', label='Consistent-P', **kwargs),
             Line(color='k', ls=':', label='Inconsistent-P', **kwargs),
             Line(color='k', marker='s', label='Galerkin PGD', **kwargs),
             Line(color='k', marker='D', label='Minimax PGD', **kwargs)]
    ax.legend(handles=lines, ncol=int(len(lines)/2), loc='upper center',
              bbox_to_anchor=(0.5, 1.45))
    

def main():
    alpha = 0.8
    fuels = ['uo2', 'mox43']
    titles = ['UO\\textsubscript{2}', '4.3\\% MOX']
    projs = ['', '-minimax']  # '' is Galerkin
    markers = ['s', 'D']
    filebase = 'Fuel_CathalauCoarseTestCriticality_{fuel}-modal{proj}.txt'
    markersize = 3.75
    kwargs = {'markevery': 2, 'markersize': markersize, 'alpha': alpha, 
              'zorder': 1.99}
    axl_prev = None
    axr_prev = None
    for i, (fuel, title) in enumerate(zip(fuels, titles)):
        axl = plt.subplot(1, len(fuels), i+1, sharex=axl_prev)
        axr = plt.twinx()
        for proj, marker in zip(projs, markers):
            filename = filebase.format(fuel=fuel, proj=proj)
            norm = plot_modal(filename, axl, axr, marker=marker, **kwargs)
        for tc in ['cp', 'ip']:
            key = f'{fuel}-{tc}'
            kwargs_tc = {'ls': '--' if tc == 'cp' else ':', 'alpha': alpha}
            draw_axhlines_l2(axl, COARSE_ERRORS[key], norm, **kwargs_tc)
            draw_axhline_dk(axr, COARSE_DK[key], **kwargs_tc)
        tick_every_decade(axl)
        tick_every_decade(axr)
        plt.title(title)
        if i == 0:
            axr.set_yticklabels([])
            axl.set_ylabel('Normalized $L^2$ Error')
        else:
            axl.set_yticklabels([])
            axr.set_ylabel('$k$-Eigenvalue Error', color='C4')
        if axr_prev:
            # Actually sharing the axes makes it hard to set tick labels
            # visible on only one plot. Just set the ylims instead.
            share_ylim(axr, axr_prev)
            share_ylim(axl, axl_prev)
        axr_prev = axr
        axl_prev = axl
    add_frame_labels(markersize=markersize)
    plt.savefig('coarse_eigen-6.pdf')


if __name__ == '__main__':
    plt.style.use('jcp.mplstyle')
    mpl.rc('figure', figsize=(6.5, 2.75))
    mpl.rc('legend', fontsize=10)
    main()
            