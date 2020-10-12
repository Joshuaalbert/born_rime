import numpy as np
import astropy.units as au
from astropy.units import Quantity
import pylab as plt
from timeit import default_timer
from matplotlib.animation import FuncAnimation

from born_rime.fourier import fourier, inv_fourier, fft_freqs
from born_rime.greens import two_dim_g, two_dim_G
from born_rime.potentials import partial_blockage, pad_with_vacuum_conditions, single_blob
from born_rime.series import born_series
from born_rime.plotting import plot_2d_image, add_colorbar

au.set_enabled_equivalencies(au.dimensionless_angles())

def plot_E_s(arrays, x, y, x0, y0, corner_indices, save_name=None):
    figsize = 6
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
    arrays = [np.abs(E_s) for E_s in arrays]
    vmin = min([E_s.min().value for E_s in arrays])
    vmax = min([E_s.max().value for E_s in arrays])

    norm = plt.Normalize(vmin, vmax)
    to_colour = lambda w: plt.cm.jet(norm(w))

    def _get_artists(artists, start):
        _, img = plot_2d_image(arrays[start], x, y, title="E_s", corner_indices=corner_indices,
                               colorizer=to_colour,ax=ax)
        sc = ax.scatter(x0[start].value, y0[start].value,c='green', label='source')
        ax.set_xlim(x.min().value, x.max().value)
        ax.set_ylim(y.min().value, y.max().value)
        ax.legend()
        artists.append(img)
        artists.append(sc)
        return artists

    def init():
        start = 0
        ax.clear()
        artists = []
        artists = _get_artists(artists, start)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.jet)
        add_colorbar(mappable, label='electric field amplitude [{}]'.format(arrays[0].unit), ax=ax)
        return artists

    def update(start):
        ax.clear()
        artists = []
        artists = _get_artists(artists, start)
        return artists

    ani = FuncAnimation(fig, update,
                        frames=range(1, len(arrays)),
                        init_func=init, blit=True)

    ani.save(save_name, fps=3.)#len(arrays) / 6.)

def test_plot():
    arrays = [np.random.uniform(size=(100,100)) for i in range(50)]
    x = np.random.uniform(size=100)
    y = np.random.uniform(size=100)
    plot_E_s(Quantity(arrays), Quantity(x), Quantity(y), Quantity(np.random.uniform(size=50)),
             Quantity(np.random.uniform(size=50)), save_name='example.mp4')

def main():
    """
    First we compare convergence of Born series to exact solution on a partially blocked bar:

    |   * source
    |
    |   _________________
    |   |   n = 1 - dn
    |   |________________
    |
    |
    |   x receiver
    |(0,0)
    """
    # x = np.linspace(-10., 10., 1000) * au.km
    # f = np.cos(2 * np.pi * x / (5. * au.km)) / au.m ** 3
    # plt.plot(x, f)
    # plt.xlabel('x [{}]'.format(x.unit))
    # plt.ylabel('y [{}]'.format(f.unit))
    # plt.show()
    #
    # F = fourier(f, x.value)
    # (s,) = fft_freqs(x)
    # plt.plot(s, np.abs(F))
    # plt.xlim(-2., 2.)
    # plt.xlabel('s [{}]'.format(s.unit))
    # plt.show()

    nu = 150e6 / au.s

    x, z, k2, k02 = partial_blockage(1000, nu, True)
    # x, z, k2, k02 = single_blob(1000, nu, 10.*
    x_medium, z_medium = x, z

    pad_size = int(x.size*0.6/2)
    k2, m, (x, z) = pad_with_vacuum_conditions(k2, k02, pad_size, x, z)

    X, Z = np.meshgrid(x, z, indexing='ij')
    g = two_dim_g(np.sqrt(k02), X, Z)

    E_s = []
    x0 = []
    z0 = []
    for i in range(50):
        _x0 = x_medium.min() + (x_medium.max() - x_medium.min())*np.sin(i/50.*np.pi/1.2)
        _z0 = z_medium.min() + (z_medium.max() - z_medium.min())*i/50.
        E_s.append(simulate_E_s(_x0, _z0, X, Z, k02, k2, x, z, g))
        x0.append(_x0)
        z0.append(_z0)
    plot_E_s(E_s, x, z, x0, z0, m, save_name='moving_point_source.mp4')

    ###
    # Plot things
    plot_2d_image(np.abs(g),x, z, title='g', corner_indices=m)
    # plot_2d_image(np.abs(results['G']),x, z, title='G',corner_indices=m)
    # plot_2d_image(np.abs(results['scatter_potential']), x, z, colorbar_name='potential',
    #               title=r'Sinusoidal partial blockage potential ($k^2(\mathbf{{x}}) - k_0^2$) at {} with boundary'.format(
    #     nu.to(au.MHz)),
    #               corner_indices=m)



    # for i, E_s in enumerate(results['E_s']):
    #     plot_2d_image(np.log(np.abs(E_s)), x, z, colorbar_name='log abs(electric field)',
    #                   title="E_s {}".format(i),
    #                   corner_indices=m)


    # E_b = E_s.T[-m[1], m[0]:-m[0]]
    # plt.plot(np.abs(E_b))
    # plt.show()
    # vis = E_b[:, None] * E_b[None,:].conj()
    # plot_2d_image(np.abs(vis), x[m[0]:-m[0]], z[m[1]:-m[1]], colorbar_name='vis amplitude')


def simulate_E_s(x0, z0, X, Z, k02, k2, x, z, g):
    R = np.sqrt((X - x0) ** 2 + (Z - z0) ** 2)
    E_i = np.exp(1j * np.sqrt(k02) * R) / (1 * au.m ** 2 + R ** 2)
    t0 = default_timer()
    E_born, results = born_series(E_i, g, k2, k02, x, z, N=3, pad=300)
    print(x0, z0, default_timer() - t0)
    return results['E_s'][-1]


if __name__ == '__main__':
    # test_plot()
    main()
