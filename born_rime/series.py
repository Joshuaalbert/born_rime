import numpy as np
from born_rime.fourier import fourier, inv_fourier
from born_rime.greens import two_dim_g

def born_series(E_i, g, k2, k02, *coords, N=1, pad=None):
    scatter_potential = k2 - k02
    G_numeric = fourier(g, *coords)
    results = dict(E_s=[], G=G_numeric, scatter_potential=scatter_potential)
    if pad is not None:
        G_numeric = np.pad(G_numeric, pad, mode='constant')
        coords = [np.pad(c, pad, mode='linear_ramp') for c in coords]
    E_born = E_i
    for i in range(1, N+1):
        source = scatter_potential * E_born
        if pad is not None:
            source = np.pad(source, pad)
        f_source = fourier(source, *coords)
        E_s = inv_fourier(G_numeric * f_source, *coords)
        if pad is not None:
            E_s = E_s[tuple([slice(pad,-pad,1)]*len(E_s.shape))]
        results['E_s'].append(E_s)
        E_born = E_i + E_s
    return E_born, results

def modified_born_series(E_i, g, k2, k02, *coords, N=1):
    k02 = 0.5*np.max(k2.real) + 0.5*np.min(k2.real)
    eps = np.max(np.abs(k2 - k02))*1.01
    V = k2 - k02 - 1j*eps

    gamma = 1j/eps * V
    if len(coords)==2:
        X,Z = np.meshgrid(coords[0] - coords[0].mean(),
                          coords[1] - coords[1].mean(),
                          indexing='ij')
        g = two_dim_g( np.sqrt(k02 + 1j * eps), X, Z)
        G_numeric = fourier(g, x, z)

def rytov_series():
    pass