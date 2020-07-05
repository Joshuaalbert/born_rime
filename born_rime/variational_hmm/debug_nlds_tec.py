"""
The problem is that before (a few days ago) this functional code seemed extremely powerful, but now suffers due
something. Perhaps it's in the BFGS implementation since few iterations seem to be solved.
"""

from born_rime.variational_hmm import TecLinearPhase, TecOnlyOriginal, TecClockLinearPhase
from born_rime.optimize import minimize
from functools import partial
from born_rime.variational_hmm.nlds_smoother import NonLinearDynamicsSmoother
from born_rime.variational_hmm.utils import constrain_std, deconstrain_std
from jax import numpy as jnp
from jax import disable_jit, jit, grad, vmap, hessian
from jax.config import config
from jax.test_util import check_grads
from jax import random

config.update("jax_enable_x64", True)


def debug_nlds_smoother():
    import numpy as onp
    import pylab as plt
    Gamma0, Omega, Sigma, T, Y_obs, amp, mu0, tec, freqs = generate_data()

    hmm = NonLinearDynamicsSmoother(TecClockLinearPhase(freqs, freeze_params=1))
    #
    # with disable_jit():
    res = jit(partial(hmm, tol=1., maxiter=200, omega_window=11, sigma_window=5, momentum=0.,
                      omega_diag_range=(0, 20.), sigma_diag_range=(0, jnp.inf)))(Y_obs, Sigma, mu0,
                                                                                                      Gamma0, Omega, amp)

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
    plt.title("omega")
    plt.show()
    plt.plot(onp.mean(onp.sqrt(onp.diagonal(res.Sigma, axis2=-2, axis1=-1)), axis=-1))
    plt.title("mean sigma")
    plt.show()


def generate_data():
    import numpy as onp
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 10. ** 2 * jnp.eye(2)
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([10., 0.01]))**2
    amp = jnp.ones_like(phase)
    return Gamma0, Omega, Sigma, T, Y_obs, amp, mu0, tec, freqs


def debug_tec_forward():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    Gamma0, Omega, Sigma, T, Y_obs, amp, hmm, mu0, tec, freqs = generate_data()
    f1 = TecOnlyOriginal(freqs)
    f2 = TecLinearPhase(freqs)



    # def sampled_var_exp(mu, gamma, y_obs, sigma, amp, S=5000000):
    #     key = random.PRNGKey(0)
    #     tec = mu + gamma * random.normal(key, (S, 1))
    #
    #     def log_prob(t):
    #         y = f1.forward_model(t, amp)
    #         lp = -0.5 * y_obs.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(sigma)) - 0.5 * jnp.sum(
    #             jnp.square((y - y_obs) / sigma))
    #         return lp
    #
    #     return jnp.mean(vmap(log_prob, 0, 0)(tec))

    for y, amp in zip(Y_obs, amp):
        # print(y)
        sigma = onp.random.uniform(0.01, 0.5, size=[freqs.size * 2])
        gamma = jnp.sqrt(jnp.diag(Gamma0))/10.#onp.random.uniform(0.1, 20., size=[1])
        mu = mu0#onp.random.uniform(-1., 1., size=[1])
        #
        # print('original', f1.var_exp(freqs, y, sigma, amp, mu[0], gamma[0]))
        # print('linear model', f2.var_exp(freqs, y, sigma, amp, mu, gamma))
        # # print('sampled',sampled_var_exp(mu[0], gamma[0], y, sigma, amp))
        # assert jnp.isclose(f1.var_exp(freqs, y, sigma, amp, mu[0], gamma[0]),
        #                    f2.var_exp(freqs, y, sigma, amp, mu, gamma)).all()
        #
        # g1 = grad(lambda mu, gamma: f1.var_exp(freqs, y, sigma, amp, mu[0], gamma[0]), argnums=[0,1])
        # g2 = grad(lambda mu, gamma: f2.var_exp(freqs, y, sigma, amp, mu, gamma), argnums=[0,1])
        # print('original grad', g1(mu, gamma), 'linearPhase g', g2(mu, gamma))
        # assert jnp.isclose(g1(mu, gamma), g2(mu, gamma)).all()

        g1 = grad(lambda _mu, _gamma: f1.neg_elbo(freqs, y, sigma, amp, _mu[0], _gamma[0], mu0, jnp.sqrt(jnp.diag(Gamma0))), argnums=[0, 1])
        g2 = grad(lambda _mu, _gamma: f2.neg_elbo(freqs, y, sigma, amp, _mu, _gamma, mu0, jnp.sqrt(jnp.diag(Gamma0))), argnums=[0, 1])
        print('original negelbo grad', g1(mu, gamma), 'linearPhase negelbo g', g2(mu, gamma))
        assert jnp.isclose(g1(mu, gamma), g2(mu, gamma)).all()


        # check_grads(lambda _mu, _gamma: f1.neg_elbo(freqs, y, sigma, amp, _mu[0], _gamma[0], mu0, jnp.sqrt(jnp.diag(Gamma0))), (mu, gamma), 3)
        # check_grads(lambda _mu, _gamma: f2.neg_elbo(freqs, y, sigma, amp, _mu, _gamma, mu0, jnp.sqrt(jnp.diag(Gamma0))), (mu, gamma), 3)
        #
        # def negelbo(params):
        #     mu, gamma = params[0], constrain_std(params[1])
        #     return f2.neg_elbo(freqs, y, sigma, amp, mu, gamma, mu0, jnp.sqrt(jnp.diag(Gamma0)))
        #
        # result = minimize(negelbo, jnp.zeros(2),method='bfgs')
        # print(result)


def debug_tec_clock_forward():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    Gamma0, Omega, Sigma, T, Y_obs, amp, hmm, mu0, tec, freqs = generate_data()
    f = TecClockLinearPhase(freqs)

    Sigma = 0.25 ** 2 * jnp.eye(48)
    Omega = jnp.diag(jnp.array([20., 0.1])) ** 2
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([100., jnp.pi])) ** 2



    for y, amp in zip(Y_obs, amp):
        sigma = jnp.sqrt(jnp.diag(Sigma))
        gamma0 = jnp.sqrt(jnp.diag(Gamma0))

        mu = mu0
        gamma = gamma0

        params = jnp.concatenate([mu, gamma])

        def negelbo(params):
            mu = params[:2]
            gamma = params[2:]
            return f.neg_elbo(freqs, y, sigma, amp, mu, gamma, mu0, gamma0)


        g = grad(negelbo)
        h = hessian(negelbo)

        print('grads', g(params))
        hess = h(params)
        print('hessian', hess)
        print('inv_hess', jnp.linalg.inv(hess))



if __name__ == '__main__':
    debug_nlds_smoother()
    # debug_tec_forward()
    # debug_tec_clock_forward()