import matplotlib

FONTSIZE = 12
FIGSIZE = (6.5, 3.5)

def set_rc(fontsize=FONTSIZE, figsize=FIGSIZE):
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('font', family='serif', serif=['Computer Modern Roman'])
    matplotlib.rc('xtick', labelsize=fontsize)
    matplotlib.rc('ytick', labelsize=fontsize)
    matplotlib.rc('legend', fontsize=fontsize)
    matplotlib.rc('legend', title_fontsize=fontsize)
    matplotlib.rc('axes', labelsize=fontsize)
    matplotlib.rc('figure', figsize=figsize)
    matplotlib.rc('savefig', transparent=True)
    matplotlib.rc('legend', framealpha=0.85)
    matplotlib.rc('axes', grid=True)
    matplotlib.rc('grid', linestyle='-', alpha=0.5)