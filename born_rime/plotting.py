import pylab as plt
from astropy.units import Quantity
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_2d_image(a, x, y, colorbar_name=None, title=None, xlabel='x', ylabel='y', cmap='bone', colorizer=None, corner_indices=None, ax=None, save_name=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
    if colorizer is None:
        norm = plt.Normalize(a.min().value, a.max().value)
        colorizer = plt.cm.get_cmap(cmap)(norm)
    img = ax.imshow(colorizer(a.T.value), interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, y.min().value, y.max().value))
    if corner_indices is not None:
        ax.plot(Quantity([x[corner_indices[0]], x[-corner_indices[0]], x[-corner_indices[0]], x[corner_indices[0]], x[corner_indices[0]]]).value,
                 Quantity([y[corner_indices[1]], y[corner_indices[1]], y[-corner_indices[1]], y[-corner_indices[1]], y[corner_indices[1]]]).value, c='red')
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel('{} [{}]'.format(xlabel, x.unit))
    if ylabel is not None:
        ax.set_ylabel('{} [{}]'.format(ylabel, y.unit))
    if colorbar_name is not None:
        plt.colorbar(label='{} [{}]'.format(colorbar_name, a.unit))
    if save_name is not None:
        plt.savefig(save_name)
    return ax, img


def add_colorbar(mappable, label, ax=None):
    if ax is None:
        last_axes = plt.gca()
        ax = mappable.axes
    else:
        last_axes = ax
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, label=label)
    fig.sca(last_axes)
    return cbar