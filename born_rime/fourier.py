import numpy as np


def fourier(a, *coords):
    """
    Evaluates

    F[a](s) = int a(x) e^{-2pi i s x} dx
    A(k ds) = sum_m a(x_m) e^{-2pi i k ds (x0 + corner_indices dx)} dx
            = e^{-2pi i k ds x0} dx sum_m a(x_m) e^{-2pi i k ds corner_indices dx}
    dx ds = 1/n => ds = 1/(dx n)
    ds x0 = k ds x0 = k/n * x0/dx
    """

    factor = fft_factor(*coords)
    return np.fft.fftshift(np.fft.fftn(a, a.shape) * factor)


def inv_fourier(A, *coords):
    factor = ifft_factor(*coords)
    return np.fft.ifftn(np.fft.ifftshift(A) * factor, A.shape)


def fft_freqs(*coords):
    s = []
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        s.append(np.fft.fftshift(np.fft.fftfreq(c.size, dx)))
    return tuple(s)


def ifft_freqs(*coords):
    s = []
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        s.append(np.fft.fftshift(np.fft.fftfreq(c.size, dx)))
    return tuple(s)


def test_coords_transformations():
    import astropy.units as au
    x = np.linspace(-10,10,101)*au.km
    (s,) = fft_freqs(x)
    _x = ifft_freqs(s)
    assert np.isclose(_x , x).all()


def fft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return np.expand_dims(t, dims)

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = np.fft.fftfreq(c.size, dx)
        log_factor =  - 2j * np.pi * s * x0
        dx_factor *= dx
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = np.exp(log_factors)*dx_factor
    return factor


def ifft_factor(*coords):
    def _add_dims(t, i):
        dims = list(range(len(coords)))
        del dims[i]
        return np.expand_dims(t, dims)

    log_factors = 0.
    dx_factor = 1.
    for i, c in enumerate(coords):
        dx = c[1] - c[0]
        x0 = c[0]
        s = np.fft.fftfreq(c.size, dx)
        dx_factor /= dx
        log_factor = 2j * np.pi * s * x0
        log_factors = log_factors + _add_dims(log_factor, i)
    factor = np.exp(log_factors)*dx_factor
    return factor


def test_fourier():
    import astropy.units as au
    def f(x):
        return np.exp(-np.pi * x.value ** 2)

    import pylab as plt
    x = np.linspace(-10., 10., 101)*au.km
    a = f(x)

    F = fourier(a, x)
    (s,) = fft_freqs(x)
    _a = inv_fourier(F, x)

    plt.plot(s, f(s), label='A true')
    plt.plot(s, np.real(F), label='A numeric')
    plt.legend()
    plt.show()

    plt.plot(x, a, label='a')
    plt.plot(x, _a, label='a rec')
    plt.legend()
    # plt.ylim(-10,3)
    plt.show()