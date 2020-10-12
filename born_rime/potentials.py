import numpy as np
from scipy.special import logsumexp, gammaln
from astropy import constants, units as au
from astropy.units import Quantity

Gauss = 1e-4 * au.T
au.set_enabled_equivalencies(au.dimensionless_angles())


def pad_with_absorbing_boundary_conditions(k2, k02, N, *coords, dn_max=0.05):
    if dn_max is None:
        dn_max = np.max(np.abs(np.sqrt(k2 / k02) - 1.))
        print("Using the dn_max={}".format(dn_max))
    alpha = np.abs(dn_max)*np.sqrt(k02)#/(np.pi*2.)
    l = N / alpha
    print("Extinction alpha={}".format(alpha))
    print("Extinction l={}".format(l))

    def log_Pn(alpha, x, N):
        log_res = -np.inf
        for n in range(N + 1):
            log_res = np.logaddexp(n * (np.log(alpha * x)) - gammaln(n + 1.), log_res)
        return np.where(x > 0, log_res, 0.)

    def evaluate_k2(alpha, x):
        k2 = k02 + alpha**2 - 2j*alpha*np.sqrt(k02)
        return k02*np.ones(x.shape)

    def _evaluate_k2(alpha, x, N):
        return alpha**2 * np.exp(np.log(N - alpha * x + 2j * np.sqrt(k02) * x) + (N - 1) * (np.log(alpha * x))
                      - log_Pn(alpha, x, N) - gammaln(N + 1.)) + k02

    def _add_other_dims(v, shape, i):
        """
        [
        Args:
            v: [D]
            shape: (s0,s1,s2,...)
            i: int

        Returns: same shape as `shape` except ith dim which is D.

        """
        dims = list(range(len(shape)))
        del dims[i]
        v = np.expand_dims(v, dims)
        grow = list(shape)
        grow[i] = 1
        return np.tile(v,grow)
    m = []
    out_coords = []
    for i,x in enumerate(coords):
        dx = x[1] - x[0]
        M = int(l / dx) + 1
        m.append(M)
        print("Dimension {} padded by {}".format(i, M))
        x_pad = np.arange(1,M+1)*dx
        k2_pad = evaluate_k2(alpha, x_pad)
        k2_before = _add_other_dims(k2_pad[::-1], k2.shape, i)
        k2_after = _add_other_dims(k2_pad, k2.shape, i)
        k2 = np.concatenate([k2_before, k2, k2_after], axis=i)
        x_out = np.concatenate([x[0] - np.arange(1,M+1)[::-1]*dx, x, x[-1]+np.arange(1,M+1)*dx])
        out_coords.append(x_out)

    return k2, m, tuple(out_coords)

def pad_with_vacuum_conditions(k2, k02, pad_size, *coords):


    def evaluate_k2(x):
        return k02*np.ones(x.shape)

    def _add_other_dims(v, shape, i):
        """
        [
        Args:
            v: [D]
            shape: (s0,s1,s2,...)
            i: int

        Returns: same shape as `shape` except ith dim which is D.

        """
        dims = list(range(len(shape)))
        del dims[i]
        v = np.expand_dims(v, dims)
        grow = list(shape)
        grow[i] = 1
        return np.tile(v,grow)
    m = []
    out_coords = []
    for i,x in enumerate(coords):
        print("Dimension {} padded by {}".format(i, pad_size))
        dx = x[1] - x[0]
        x_pad = np.arange(1,pad_size+1)*dx
        k2_pad = evaluate_k2(x_pad)
        m.append(pad_size)
        k2_before = _add_other_dims(k2_pad[::-1], k2.shape, i)
        k2_after = _add_other_dims(k2_pad, k2.shape, i)
        k2 = np.concatenate([k2_before, k2, k2_after], axis=i)
        x_out = np.concatenate([x[0] - np.arange(1,pad_size+1)[::-1]*dx, x, x[-1]+np.arange(1, pad_size+1)*dx])
        out_coords.append(x_out)

    return k2, m, tuple(out_coords)

def appleton_hartree(ne, nu):
    def _plasma_freqency_squared(fed):
        omega_p_squared = fed * (constants.e.si ** 2 / constants.eps0 / constants.m_e)
        return omega_p_squared

    omega_0_squared = _plasma_freqency_squared(ne)
    dn = omega_0_squared / (2 * np.pi * nu) ** 2
    return 1. - dn

def partial_blockage(N, nu, sinusoidal_blockage=False):
    """
    |   * source
    |
    |   _________________
    |   |   n = 1 - dn
    |   |________________
    |
    |
    |   x receiver
    |(0,0)

    Args:
        x:
        z:
        nu:

    Returns:

    """

    ne = 2e12 / au.m ** 3
    wavelength = constants.c.si / nu
    x = np.arange(-N//2, N-N//2,1) * 0.25 * wavelength
    z = np.arange(-N//2, N-N//2,1) * 0.25 * wavelength

    n_ionosphere = appleton_hartree(ne, nu)

    k0 = 2. * np.pi / wavelength
    X, Z = np.meshgrid(x, z, indexing='ij')

    z_bar_bottom = z.min() + 0.5 * (z.max() - z.min())
    z_bar_top = z_bar_bottom + 10. * wavelength
    x_bar_left = x.min() + 0. * (x.max() - x.min())

    where_bar = (X > x_bar_left) & (Z > z_bar_bottom) & (Z < z_bar_top)
    if sinusoidal_blockage:
        refractive_index = np.where(where_bar, 1. - (1. - n_ionosphere) * np.cos(2 * np.pi * X / (10. * wavelength)),
                                    1.)
    else:
        refractive_index = np.where(where_bar, n_ionosphere, 1.)

    k2 = 4. * np.pi ** 2 * refractive_index ** 2 / wavelength ** 2
    return x, z, k2, k0 ** 2

def single_blob(N, nu, l):
    """
    |   * source
    |
    |   _________________
    |   |   n = 1 - dn
    |   |________________
    |
    |
    |   x receiver
    |(0,0)

    Args:
        x:
        z:
        nu:

    Returns:

    """

    ne = 2e12 / au.m ** 3
    wavelength = constants.c.si / nu
    x = np.arange(-N//2, N-N//2,1) * 0.25 * wavelength
    z = np.arange(-N//2, N-N//2,1) * 0.25 * wavelength

    n_ionosphere = appleton_hartree(ne, nu)

    k0 = 2. * np.pi / wavelength
    X, Z = np.meshgrid(x, z, indexing='ij')

    z_blob = z.min() + 0.5 * (z.max() - z.min())
    x_blob = x.min() + 0.5 * (x.max() - x.min())

    refractive_index = (n_ionosphere - 1) * np.exp(-0.5*((X-x_blob)**2 + (Z-z_blob)**2)/l**2) + 1.

    k2 = 4. * np.pi ** 2 * refractive_index ** 2 / wavelength ** 2
    return x, z, k2, k0 ** 2


def test_partial_blockage():
    import pylab as plt
    nu = 100e6 / au.s
    N = 1000

    x, z, k2, k02 = partial_blockage(N, nu)
    scattering_potential = k2 - k02

    plt.imshow(scattering_potential.T.value, interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')
    plt.title(r'Partial blockage potential ($k^2(\mathbf{{x}}) - k_0^2$) at {}'.format(nu.to(au.MHz)))
    plt.colorbar(label='potential [{}]'.format(scattering_potential.unit))
    plt.show()

    x, z, k2, k02 = partial_blockage(N, nu, sinusoidal_blockage=True)
    scattering_potential = k2 - k02

    plt.imshow(scattering_potential.T.value, interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')
    plt.title(r'Sinusoidal partial blockage potential ($k^2(\mathbf{{x}}) - k_0^2$) at {}'.format(nu.to(au.MHz)))
    plt.colorbar(label='potential [{}]'.format(scattering_potential.unit))
    plt.show()

    k2, m, (x,z) = pad_with_absorbing_boundary_conditions(k2, k02, 4, x, z, dn_max=0.01)
    scattering_potential = k2 - k02

    plt.imshow(np.abs(scattering_potential.T.value), interpolation='nearest', origin='lower',
               extent=(x.min().value, x.max().value, z.min().value, z.max().value),
               cmap='bone')
    print(x)
    plt.plot(Quantity([x[m[0]], x[-m[0]], x[-m[0]], x[m[0]], x[m[0]]]).value, Quantity([z[m[1]], z[m[1]],z[-m[1]],z[-m[1]],z[m[1]]]).value, c='red')
    plt.title(r'Sinusoidal partial blockage potential ($k^2(\mathbf{{x}}) - k_0^2$) at {} with boundary'.format(nu.to(au.MHz)))
    plt.colorbar(label='potential [{}]'.format(scattering_potential.unit))
    plt.show()
