from pylab import *
import scipy.stats as ss
import seaborn as sns

def kdeplot_2d_clevels(xs, ys, levels=11, **kwargs):
    try:
        len(levels)
        f = 1 - np.array(levels)
    except TypeError:
        f = linspace(0, 1, levels)[1:-1]
    k = ss.gaussian_kde(row_stack((xs, ys)))
    size = max(10*(len(f)+2), 500)
    c = np.random.choice(len(xs), size=size)
    p = k(row_stack((xs[c], ys[c])))
    i = argsort(p)
    l = array([p[i[int(round(ff*len(i)))]] for ff in f])

    Dx = np.percentile(xs, 99) - np.percentile(xs, 1)
    Dy = np.percentile(ys, 99) - np.percentile(ys, 1)

    x = linspace(np.percentile(xs, 1)-0.1*Dx, np.percentile(xs, 99)+0.1*Dx, 128)
    y = linspace(np.percentile(ys, 1)-0.1*Dy, np.percentile(ys, 99)+0.1*Dy, 128)

    XS, YS = meshgrid(x, y, indexing='ij')
    ZS = k(row_stack((XS.flatten(), YS.flatten()))).reshape(XS.shape)

    if 'cmap' not in kwargs:
        line, = plot([], [])
        kwargs['cmap'] = sns.dark_palette(line.get_color(), as_cmap=True)

    ax = kwargs.pop('ax', gca())

    ax.contour(XS, YS, ZS, levels=l, **kwargs)
