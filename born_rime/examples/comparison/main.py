import numpy as np
import astropy.units as au
from astropy.units import Quantity
import pylab as plt

from born_rime.fourier import fourier, inv_fourier, fft_freqs
from born_rime.greens import two_dim_g, two_dim_G
from born_rime.potentials import partial_blockage, pad_with_absorbing_boundary_conditions


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

    nu = 50e6 / au.s

    x, z, k2, k02 = partial_blockage(1000, nu, True)
    k2, m, (x, z) = pad_with_absorbing_boundary_conditions(k2, k02, 1, x, z, dn_max=0.001)
    # corner_indices = [0,0]
    scatter_potential = (k2 - k02)/k02

    plt.imshow(np.abs(scatter_potential.T.value), interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')

    plt.plot(Quantity([x[m[0]], x[-m[0]], x[-m[0]], x[m[0]], x[m[0]]]).value,
             Quantity([z[m[1]], z[m[1]], z[-m[1]], z[-m[1]], z[m[1]]]).value, c='red')
    plt.title(r'Sinusoidal partial blockage potential ($k^2(\mathbf{{x}}) - k_0^2$) at {} with boundary'.format(
        nu.to(au.MHz)))
    plt.colorbar(label='potential [{}]'.format(scatter_potential.unit))
    plt.show()

    X,Z = np.meshgrid(x,z,indexing='ij')
    R = np.sqrt((X-(-300.*au.m))**2 + (Z-(0*au.m))**2)
    E_i = np.exp(1j*np.sqrt(k02)*R)/(1*au.m**2 + R**2)
    E_i = np.exp(1j*np.sqrt(k02)*Z)

    g = two_dim_g(np.sqrt(k02), X, Z)

    plt.imshow((np.abs(g).value), interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')
    plt.colorbar()
    plt.plot(Quantity([x[m[0]], x[-m[0]], x[-m[0]], x[m[0]], x[m[0]]]).value,
             Quantity([z[m[1]], z[m[1]], z[-m[1]], z[-m[1]], z[m[1]]]).value, c='red')
    plt.title('g')
    plt.show()


    G_numeric = fourier(g, x, z)
    # sx, sy = fft_freqs(x,z)
    # Sx, Sy = np.meshgrid(sx,sy, indexing='ij')
    # G_numeric = two_dim_G(np.sqrt(k02),Sx, Sy)
    n = G_numeric.shape[0]
    pad_size = 1#int((n*0.6)/2.)

    plt.imshow((np.abs(G_numeric).value), interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')
    plt.colorbar()
    plt.plot(Quantity([x[m[0]], x[-m[0]], x[-m[0]], x[m[0]], x[m[0]]]).value,
             Quantity([z[m[1]], z[m[1]], z[-m[1]], z[-m[1]], z[m[1]]]).value, c='red')
    plt.title('G')
    plt.show()

    G_padded = np.pad(G_numeric,pad_size, mode='constant')
    x_padded = np.pad(x, pad_size, mode='linear_ramp')
    z_padded = np.pad(z, pad_size, mode='linear_ramp')

    E_born = E_i
    for i in range(1, 4):
        source = scatter_potential * E_born
        source_padded = np.pad(source, pad_size)
        f_source_padded = fourier(source_padded, x_padded, z_padded)
        E_born = E_i + k02*inv_fourier(G_padded * f_source_padded, x_padded, z_padded)[pad_size:-pad_size,pad_size:-pad_size]
        E_s = E_born - E_i

        plt.imshow((np.abs(E_s.T).value), interpolation='nearest', origin='lower',
                   extent=(x.min().value, x.max().value, z.min().value, z.max().value),
                   cmap='bone')
        plt.colorbar()
        plt.plot(Quantity([x[m[0]], x[-m[0]], x[-m[0]], x[m[0]], x[m[0]]]).value,
                 Quantity([z[m[1]], z[m[1]], z[-m[1]], z[-m[1]], z[m[1]]]).value, c='red')
        plt.title('Born-{}'.format(i))
        plt.show()

    # plt.plot(x, np.abs(E_born.T[0,:]))
    # # plt.xscale('log')
    # plt.show()
    #
    # plt.plot(x, np.angle(E_born.T[0, :]))
    # plt.show()

    _vis = E_born.T[200, :, None] * E_born.T[200, None, :].conj()
    vis = [np.mean(np.diagonal(_vis, i)) for i in range(x.size)]
    b = x[:, None] - x[None, :]
    plt.plot(b[0, :], np.abs(vis))
    plt.title('|vis|')
    plt.show()
    plt.plot(b[0, :], np.angle(vis))
    plt.title('Arg(vis)')
    plt.show()
    pass


if __name__ == '__main__':
    main()
