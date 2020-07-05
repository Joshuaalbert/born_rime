from jax.test_util import check_grads

from born_rime.variational_hmm import TecLinearPhaseLinearised, TecLinearPhase, TecClockLinearPhaseLinearised, TecClockLinearPhase, TecOnlyOriginal
from functools import partial
from born_rime.variational_hmm.utils import constrain_sigma, deconstrain_sigma, windowed_mean, mvn_kl, \
    scalar_KL, fill_triangular, fill_triangular_inverse, polyfit
from born_rime.variational_hmm.nlds_smoother import NonLinearDynamicsSmoother
from jax import numpy as jnp, numpy as np
from jax import jit, vmap, grad
import jax
from jax import random, disable_jit, jit
from jax.config import config

config.update("jax_enable_x64", True)


def test_batched_multi_dot():
    batched_multi_dot = vmap(jnp.linalg.multi_dot, 0, 0)
    assert (5, 4, 2) == batched_multi_dot([jnp.ones((5, 4, 3)),
                                           jnp.ones((5, 3, 3)),
                                           jnp.ones((5, 3, 2))]).shape


def test_soft_pmap():
    def fun(x, z):
        return (x - z, x + z)

    x = jnp.ones((24, 5))
    z = jnp.array([1.])

    pfun = jax.soft_pmap(partial(fun, z=z), in_axes=(0,))
    print(pfun(x))


def test_pmap():
    with disable_jit():
        def fun(x, z):
            print(x.shape, z.shape)
            return (x - z, x + z)

        x = jnp.ones((12, 5))
        z = jnp.array([1.])

        pfun = jax.pmap(fun, in_axes=(0, None))
        pfun(x, z)


def test_constrain():
    assert jnp.isclose(0.5, constrain_sigma(deconstrain_sigma(0.5)))
    assert jnp.isclose(0.01, constrain_sigma(deconstrain_sigma(0.01)))
    assert jnp.isclose(0.01, constrain_sigma(deconstrain_sigma(0.001)))



def test_nlds_smoother():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(0.*onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    const = 0.5*onp.linspace(-onp.pi, onp.pi, T)
    print("Const change {}".format(jnp.sqrt(jnp.mean(jnp.diff(const)**2))))
    phase = tec[:, None] / freqs * TEC_CONV + const[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.5 * onp.random.normal(size=Y.shape)
    # Y_obs[100::6, :] += 3. * onp.random.normal(size=Y[100::6, :].shape)
    # Y_obs[1::6, :] += 3. * onp.random.normal(size=Y[1::6, :].shape)
    # hmm = NonLinearDynamicsSmoother(TecClockLinearPhaseLinearised(freqs, tol=[0.5, 0.01], maxiter=2, momentum=0.))
    hmm = NonLinearDynamicsSmoother(TecClockLinearPhase(freqs))

    Sigma = 0.25 ** 2 * jnp.eye(48)
    Omega = jnp.diag(jnp.array([20.,0.1]))**2
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([100.,np.pi]))**2
    amp = jnp.ones_like(phase)

    # prior_Gamma, post_mu_f, post_Gamma_f = hmm.forward_filter(Y_obs, jnp.tile(Sigma[None, :,:], [T,1,1]), mu0, Gamma0,
    #                            jnp.tile(Omega[None, :,:], [T,1,1]), amp)
    # # print(post_mu_f[:,0])
    # plt.plot(post_mu_f[:,0])
    # plt.title('tec f')
    # plt.show()
    # plt.plot(post_mu_f[:, 1])
    # plt.title('const f')
    # plt.show()
    # plt.plot(jnp.sqrt(post_Gamma_f[:,0,0]))
    # plt.title('uncert tec f')
    # plt.show()
    # post_mu_b, post_Gamma_b, inter_Gamma = hmm.backward_filter(prior_Gamma, post_mu_f, post_Gamma_f)
    # # print(post_mu_b)
    # plt.plot(post_mu_b[:, 0])
    # plt.title('tec b')
    # plt.show()
    # plt.plot(jnp.sqrt(post_Gamma_b[:,0,0]))
    # plt.title('uncert tec b')
    # plt.show()
    # plt.plot(post_mu_b[:, 1])
    # plt.title('const b')
    # plt.show()
    # plt.plot(jnp.sqrt(post_Gamma_b[:, 1, 1]))
    # plt.title('uncert const b')
    # plt.show()
    # mu0_i, Gamma0_i, Sigma_i, Omega_i = hmm.parameter_estimation(post_mu_b, post_Gamma_b, inter_Gamma, Y,
    #                                                               amp, omega_window=1,
    #                                                               sigma_window=1)
    # print(mu0_i)
    # return
    with disable_jit():
        res = jit(partial(hmm, tol=[0.5, 0.01], maxiter=1, omega_diag_range=((0.01,0.001),(20., 0.1)),
                          omega_window=None, sigma_window=None, momentum=[0., 0., 0., 0.], beg=100, quack=25))(
            Y_obs, Sigma, mu0, Gamma0, Omega, amp)
    # res = jit(partial(hmm2, tol=0., maxiter=5, omega_window=11, sigma_window=11, momentum=1.))(Y_obs, res.Sigma, res.mu0,
    #                                                                                            res.Gamma0, res.Omega, amp)
    assert res.post_mu.shape[0] == T
    print(res.converged, res.niter)
    plt.plot(tec, label='true tec')
    plt.plot(res.post_mu[:, 0], label='infer tec')
    plt.fill_between(onp.arange(T),
                     res.post_mu[:, 0] - onp.sqrt(res.post_Gamma[:, 0, 0]),
                     res.post_mu[:, 0] + onp.sqrt(res.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.legend()
    plt.show()

    plt.plot(onp.sqrt(res.post_Gamma[:, 0, 0]))
    plt.title("Uncertainty tec")
    plt.show()

    plt.plot(tec - res.post_mu[:, 0], label='infer')
    plt.fill_between(onp.arange(T),
                     (tec - res.post_mu[:, 0]) - onp.sqrt(res.post_Gamma[:, 0, 0]),
                     (tec - res.post_mu[:, 0]) + onp.sqrt(res.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.title("Residual tec")
    plt.legend()
    plt.show()
    plt.plot(jnp.sqrt(res.Omega[:, 0, 0]))
    plt.title("omega tec")
    plt.show()
    plt.plot(onp.mean(onp.sqrt(onp.diagonal(res.Sigma, axis2=-2, axis1=-1)), axis=-1))
    plt.title("mean sigma")
    plt.show()
    plt.plot(res.post_mu[:,1])
    plt.title('const')
    plt.show()
    plt.plot(onp.sqrt(res.post_Gamma[:, 1, 1]))
    plt.title("Uncertainty const")
    plt.show()
    plt.plot(jnp.sqrt(res.Omega[:, 1, 1]))
    plt.title("omega const")
    plt.show()


def test_tec_forward():
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
    amp = jnp.ones_like(phase)
    f1 = TecOnlyOriginal(freqs)
    f2 = TecLinearPhase(freqs)


    # for t in tec:
    #     t = jnp.array([t])
    #     assert jnp.isclose(f1.forward_model(t, jnp.ones(freqs.size)),
    #                        f2.forward_model(t, jnp.ones(freqs.size))).all()

    from jax import random

    def sampled_var_exp(mu, gamma, y_obs, sigma, amp, S=5000000):
        key = random.PRNGKey(0)
        tec = mu + gamma * random.normal(key, (S, 1))

        def log_prob(t):
            y = f1.forward_model(t, amp)
            lp = -0.5 * y_obs.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(sigma)) - 0.5 * jnp.sum(
                jnp.square((y - y_obs) / sigma))
            return lp

        return jnp.mean(vmap(log_prob, 0, 0)(tec))

    for y, amp in zip(Y_obs, amp):
        sigma = onp.random.uniform(0.01, 0.5, size=[freqs.size * 2])
        gamma = onp.random.uniform(0.1, 20., size=[1])
        mu = onp.random.uniform(-100., 100., size=[1])
        print('original', f1.var_exp(freqs, y, sigma, amp, mu[0], gamma[0]))
        print('linear model', f2.var_exp(freqs, y, sigma, amp, mu, gamma))
        print('sampled',sampled_var_exp(mu[0], gamma[0], y, sigma, amp))
        assert jnp.isclose(f1.var_exp(freqs, y, sigma, amp, mu[0], gamma[0]),
                           f2.var_exp(freqs, y, sigma, amp, mu, gamma)).all()
        g1 = grad(lambda mu, gamma: f1.var_exp(freqs, y, sigma, amp, mu, gamma))
        g2 = grad(lambda mu, gamma: f2.var_exp(freqs, y, sigma, amp, mu, gamma))
        assert jnp.isclose(g1(mu), g2(mu)).all()


def test_tec_only_linear_phase():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV #+ onp.linspace(0., np.pi, T)[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    hmm1 = NonLinearDynamicsSmoother(TecLinearPhase(freqs))
    hmm2 = NonLinearDynamicsSmoother(TecLinearPhaseLinearised(freqs))

    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    amp = jnp.ones_like(phase)

    res1 = jit(partial(hmm1, tol=1., maxiter=None, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
                                                                                                  Gamma0, Omega, amp)

    res2 = jit(partial(hmm2, tol=1., maxiter=None, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
                                                                                                  Gamma0, Omega, amp)
    # assert jnp.all(jnp.isclose(res1.post_mu, res2.post_mu))
    print(res1.converged, res1.niter)
    print(res2.converged, res2.niter)
    plt.plot(tec, label='true')
    plt.plot(res1.post_mu[:, 0], label='infer1')
    plt.fill_between(onp.arange(T),
                     res1.post_mu[:, 0] - onp.sqrt(res1.post_Gamma[:, 0, 0]),
                     res1.post_mu[:, 0] + onp.sqrt(res1.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.plot(res2.post_mu[:, 0], label='infer2')
    plt.fill_between(onp.arange(T),
                     res2.post_mu[:, 0] - onp.sqrt(res2.post_Gamma[:, 0, 0]),
                     res2.post_mu[:, 0] + onp.sqrt(res2.post_Gamma[:, 0, 0]),
                     alpha=0.5)
    plt.show()

    plt.plot(res1.post_mu[:, 0] - res2.post_mu[:, 0])
    plt.show()


def test_tec_clock_linear_phase():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    const = onp.linspace(0., 2. * onp.pi, T)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV + const[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    hmm2 = NonLinearDynamicsSmoother(TecLinearPhase(freqs))
    # hmm2 = NonLinearDynamicsSmoother(TecClockLinearPhase(freqs))

    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    amp = jnp.ones_like(phase)

    # res1 = jit(partial(hmm1, tol=1., maxiter=15, omega_window=10, sigma_window=10, momentum=0.1))(Y_obs, Sigma, mu0,
    #                                                                                             Gamma0, Omega, amp)

    Omega = jnp.diag(jnp.array([1., 1.]))
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([100., 2.])) ** 2

    res2 = jit(partial(hmm2, tol=[1., 1.], maxiter=None, omega_window=10, sigma_window=10, momentum=0.5))(Y_obs, Sigma,
                                                                                                          mu0,
                                                                                                          Gamma0, Omega,
                                                                                                          amp)
    # assert jnp.all(jnp.isclose(res1.post_mu, res2.post_mu))
    # print(res1.converged, res1.niter)
    print(res2.converged, res2.niter)
    print(res2)
    # plt.plot(res1.post_mu[:, 0], label='infer')
    plt.plot(res2.post_mu[:, 0], label='infer')
    plt.plot(tec)
    plt.legend()
    plt.show()

    plt.plot(res2.post_mu[:, 1] / 100.)
    plt.plot(const)
    plt.show()


def test_nlds_smoother_pmap():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    M = 24
    tec = jnp.cumsum(10. * onp.random.normal(size=[M, T]), axis=-1)
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[..., None] / freqs * TEC_CONV
    Y = onp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.5 * onp.random.normal(size=Y.shape)
    Y_obs[:, 500:550:2, :] += 3. * onp.random.normal(size=Y[:, 500:550:2, :].shape)
    hmm = NonLinearDynamicsSmoother(TecOnlyOriginal(freqs))

    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 1. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
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
                        in_axes=(0, 0))(Y_obs, amp))


def test_windowed_mean():
    a = jnp.arange(5 * 2).reshape(5, 2)
    assert windowed_mean(a, 2).shape == (5, 2)
    assert windowed_mean(a, 3).shape == (5, 2)
    a = jnp.arange(3)
    assert jnp.all(windowed_mean(a, 1) == a)
    b = jnp.array([1 + 0 + 1, 0 + 1 + 2, 1 + 2 + 1]) / 3.
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
    c1 = onp.polyfit(x, y, deg=2)
    c2 = polyfit(x, y, deg=2)
    assert jnp.isclose(c1, c2).all()
