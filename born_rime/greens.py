import numpy as np
import pylab as plt
from scipy.special import hankel1
from born_rime.fourier import fourier, ifft_freqs, inv_fourier


def _get_dx(X, i):
    i0 = [0] * len(X.shape)
    i0[i] = 0
    i1 = [0] * len(X.shape)
    i1[i] = 1
    return X[tuple(i1)] - X[tuple(i0)]

def two_dim_g(k0, X, Y):
    r = np.sqrt(X**2 + Y**2)
    g = 0.25j * hankel1(0., k0 * r)
    max_pix = 0j
    dx = _get_dx(X,0)
    dy = _get_dx(Y,1)
    for _ in range(100):
        r = np.sqrt((dx*np.random.uniform(-0.5, 0.5)) ** 2 + (dy*np.random.uniform(-0.5, +0.5)) ** 2)
        max_pix += 0.25j * hankel1(0., k0 * r)
    max_pix = max_pix/100.
    g = np.where(np.isnan(g) | np.isinf(g), max_pix, g)
    return g


def n_dim_G(k0, *S):
    k2 = 4.*np.pi**2 * sum([s**2 for s in S])
    diff = (k2 - k0**2)
    eps = 4.*np.pi**2 * sum([(5.*_get_dx(s, i))**2 for i, s in enumerate(S)])#empirical fudge
    return diff/(diff**2 + eps**2)


def two_dim_G(k0, Sx, Sy):
    return n_dim_G(k0,Sx, Sy)

def test_two_dim_greens():
    wavelength = 1.
    km = 2.*np.pi/wavelength
    x = np.linspace(-100.,100., 1001) * wavelength
    X,Y = np.meshgrid(x,x,indexing='ij')
    g = two_dim_g(km, X,Y)
    plt.imshow(np.abs(g))
    plt.colorbar()
    plt.title("g_true abs")
    plt.show()
    G_numeric = fourier(g, x, x)
    plt.imshow(np.abs(G_numeric))
    plt.colorbar()
    plt.title("G_num abs")
    plt.show()
    sx,sy = ifft_freqs(x, x)
    Sx, Sy = np.meshgrid(sx,sy,indexing='ij')
    G_analytic = two_dim_G(km,Sx,Sy)
    plt.imshow(np.abs(G_analytic))
    plt.colorbar()
    plt.title("G_true abs")
    plt.show()

    g_num = inv_fourier(G_analytic, x, x)
    plt.imshow(np.abs(g_num))
    plt.colorbar()
    plt.title('g_num abs')
    plt.show()