from jax.test_util import check_grads

import born_rime.variational_hmm
from functools import partial
from born_rime.variational_hmm.utils import constrain_sigma, deconstrain_sigma, windowed_mean, mvn_kl, \
    scalar_KL, fill_triangular, fill_triangular_inverse, polyfit
from born_rime.variational_hmm.forward_update import ForwardUpdateEquation, TecAmpsDiagSigmaDiagOmega,\
    TecOnlyAmpDiagLinearPhaseDiagSigma, TecClockAmpDiagLinearPhaseDiagSigma
from born_rime.variational_hmm.nlds_smoother import NonLinearDynamicsSmoother
from jax import numpy as jnp, numpy as np
from jax import jit, vmap
import jax
from jax import random, disable_jit, jit
from jax.config import config
config.update("jax_enable_x64", True)

def test_batched_multi_dot():
    batched_multi_dot = vmap(jnp.linalg.multi_dot, 0, 0)
    assert (5,4,2) == batched_multi_dot([jnp.ones((5,4,3)),
                             jnp.ones((5,3,3)),
                             jnp.ones((5,3,2))]).shape

def test_soft_pmap():
    def fun(x,z):
        return (x-z, x+z)
    x = jnp.ones((24,5))
    z = jnp.array([1.])

    pfun = jax.soft_pmap(partial(fun, z=z),in_axes=(0,))
    print(pfun(x))

def test_pmap():
    with disable_jit():
        def fun(x,z):
            print(x.shape, z.shape)
            return (x-z, x+z)
        x = jnp.ones((12,5))
        z = jnp.array([1.])

        pfun = jax.pmap(fun,in_axes=(0,None))
        pfun(x,z)

class TestForwardUpdateEquation(ForwardUpdateEquation):
    def __init__(self,N):
        self.N = N

    def forward_model(self, mu, *control_params):
        return jnp.mean(mu) * jnp.ones(self.N)

def test_empty_run():
    with disable_jit():
        empty_forward = TestForwardUpdateEquation(10)
        model = NonLinearDynamicsSmoother(empty_forward)
        key = random.PRNGKey(0)
        T = 100
        Y = random.normal(key, shape=[T, 10])
        Sigma = random.normal(key, shape=[T, 10, 10])
        mu0 = random.normal(key, shape=[6])
        Gamma0 = random.normal(key, shape=[6,6])
        Omega = random.normal(key, shape=[6,6])

        results = jit(model)(Y, Sigma, mu0, Gamma0, Omega)
        assert results.post_mu.dtype == mu0.dtype
        assert results.converged

        # print(results)


def test_constrain():
    assert jnp.isclose(0.5, constrain_sigma(deconstrain_sigma(0.5)))
    assert jnp.isclose(0.01, constrain_sigma(deconstrain_sigma(0.01)))
    assert jnp.isclose(0.01, constrain_sigma(deconstrain_sigma(0.001)))


def test_nlds_smoother():

    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2,:] += 3. * onp.random.normal(size=Y[500:550:2,:].shape)
    hmm = NonLinearDynamicsSmoother(TecAmpsDiagSigmaDiagOmega(freqs))

    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1.**2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100.**2 * jnp.eye(1)
    amp = jnp.ones_like(phase)
    
    res = jit(partial(hmm, tol=1., maxiter=15, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0, Gamma0, Omega, amp)
    print(res.Sigma[0,:,:])
    assert res.post_mu.shape[0] == T
    print(res.converged, res.niter)
    plt.plot(tec, label='true')
    plt.plot(res.post_mu[:,0], label='infer')
    plt.fill_between(onp.arange(T),
                     res.post_mu[:,0]-onp.sqrt(res.post_Gamma[:,0,0]),
                     res.post_mu[:,0]+onp.sqrt(res.post_Gamma[:,0,0]),
                     alpha=0.5)
    plt.show()

    plt.plot(onp.sqrt(res.post_Gamma[:,0,0]))
    plt.show()

    plt.plot(tec- res.post_mu[:, 0], label='infer')
    plt.fill_between(onp.arange(T),
                     (tec- res.post_mu[:, 0]) - onp.sqrt(res.post_Gamma[:, 0, 0]),
                     (tec- res.post_mu[:, 0]) + onp.sqrt(res.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.legend()
    plt.show()
    plt.plot(jnp.sqrt(res.Omega[:,0,0]))
    plt.show()
    plt.plot(onp.mean(onp.sqrt(onp.diagonal(res.Sigma, axis2=-2, axis1=-1)), axis=-1))
    plt.show()
    
def test_tec_only_linear_phase():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    hmm1 = NonLinearDynamicsSmoother(TecOnlyAmpDiagLinearPhaseDiagSigma(freqs))
    hmm2 = NonLinearDynamicsSmoother(TecAmpsDiagSigmaDiagOmega(freqs))


    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    amp = jnp.ones_like(phase)

    res1 = jit(partial(hmm1, tol=1., maxiter=15, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
                                                                                                Gamma0, Omega, amp)

    res2 = jit(partial(hmm2, tol=1., maxiter=15, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
                                                                                                  Gamma0, Omega, amp)
    # assert jnp.all(jnp.isclose(res1.post_mu, res2.post_mu))
    print(res1.converged, res1.niter)
    print(res2.converged, res2.niter)
    plt.plot(tec, label='true')
    plt.plot(res1.post_mu[:, 0], label='infer')
    plt.fill_between(onp.arange(T),
                     res1.post_mu[:, 0] - onp.sqrt(res1.post_Gamma[:, 0, 0]),
                     res1.post_mu[:, 0] + onp.sqrt(res1.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.plot(res2.post_mu[:, 0], label='infer')
    plt.fill_between(onp.arange(T),
                     res2.post_mu[:, 0] - onp.sqrt(res2.post_Gamma[:, 0, 0]),
                     res2.post_mu[:, 0] + onp.sqrt(res2.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.show()

    plt.plot(res1.post_mu[:,0] - res2.post_mu[:,0])
    plt.show()


def test_tec_clock_linear_phase():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    const = onp.linspace(0., 2.*onp.pi,T)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV + const[:,None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    hmm1 = NonLinearDynamicsSmoother(TecOnlyAmpDiagLinearPhaseDiagSigma(freqs))
    hmm2 = NonLinearDynamicsSmoother(TecClockAmpDiagLinearPhaseDiagSigma(freqs))


    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    amp = jnp.ones_like(phase)

    # res1 = jit(partial(hmm1, tol=1., maxiter=15, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
    #                                                                                             Gamma0, Omega, amp)

    Omega = jnp.diag(jnp.array([1., 1.]))
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([100., 2.]))**2

    res2 = jit(partial(hmm2, tol=[1., 1.], maxiter=None, omega_window=10, sigma_window=10, momentum=0.5))(Y_obs, Sigma, mu0,
                                                                                                  Gamma0, Omega, amp)
    # assert jnp.all(jnp.isclose(res1.post_mu, res2.post_mu))
    # print(res1.converged, res1.niter)
    print(res2.converged, res2.niter)
    print(res2)
    # plt.plot(res1.post_mu[:, 0], label='infer')
    plt.plot(res2.post_mu[:, 0], label='infer')
    plt.plot(tec)
    plt.legend()
    plt.show()

    plt.plot(res2.post_mu[:,1]/100.)
    plt.plot(const)
    plt.show()
    

def test_nlds_smoother_pmap():

    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    M = 24
    tec = jnp.cumsum(10. * onp.random.normal(size=[M,T]), axis=-1)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[..., None] / freqs * TEC_CONV
    Y = onp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.5 * onp.random.normal(size=Y.shape)
    Y_obs[:, 500:550:2,:] += 3. * onp.random.normal(size=Y[:,500:550:2,:].shape)
    hmm = NonLinearDynamicsSmoother(TecAmpsDiagSigmaDiagOmega(freqs))

    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1.**2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100.**2 * jnp.eye(1)
    amp = jnp.ones_like(phase)

    def constant_folded(Sigma, mu0, Gamma0, Omega,
                        tol=1., maxiter=10, omega_window=10, sigma_window=10):
        @jit
        def f(Y_obs, amp):
            return hmm(Y_obs, Sigma, mu0, Gamma0, Omega, amp,
                       tol=tol, maxiter=maxiter,
                       omega_window=omega_window, sigma_window=sigma_window)
        return f
    print(jax.soft_pmap(constant_folded(Sigma, mu0, Gamma0, Omega,
                                         tol=1., maxiter=10, omega_window=10, sigma_window=10),
                    in_axes=(0,0))(Y_obs, amp))


def test_windowed_mean():
    a = jnp.arange(5*2).reshape(5,2)
    assert windowed_mean(a, 2).shape == (5,2)
    assert windowed_mean(a, 3).shape == (5,2)
    a = jnp.arange(3)
    assert jnp.all(windowed_mean(a, 1) == a)
    b = jnp.array([1+0+1, 0+1+2, 1+2+1])/3.
    assert jnp.all(windowed_mean(a, 3) == b)


def test_mvn_kl():
    mu_a = np.ones(2)
    Cov_a = np.eye(2) * 1. ** 2
    mu_b = np.zeros(2)
    Cov_b = np.eye(2) * 2. ** 2

    check_grads(mvn_kl, (mu_a, np.sqrt(Cov_a), mu_b, np.sqrt(Cov_b)), order=2)

    assert np.isclose(mvn_kl(mu_a, np.sqrt(Cov_a), mu_b, np.sqrt(Cov_b)),
                      scalar_KL(mu_a, np.diag(np.sqrt(Cov_a)), mu_b, np.diag(np.sqrt(Cov_b))))


def test_fill_triangular():
    x = np.arange(15) + 1
    xc = np.concatenate([x, x[5:][::-1]])
    y = np.reshape(xc, [5, 5])
    y = np.triu(y, k=0)
    assert np.all(y == fill_triangular(x, upper=True))

    assert np.all(fill_triangular_inverse(y, upper=True) == x)


def test_polyfit():
    import numpy as onp
    x = onp.random.normal(size=5)
    y = onp.random.normal(size=5)
    c1 = onp.polyfit(x,y,deg=2)
    c2 = polyfit(x,y,deg=2)
    assert jnp.isclose(c1,c2).all()