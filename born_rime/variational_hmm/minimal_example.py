import born_rime.variational_hmm
from born_rime.optimize import minimize
from born_rime.variational_hmm.utils import constrain_std, deconstrain_std, scalar_KL, value_and_jacobian
from jax import numpy as jnp
from jax import jit

from born_rime.variational_hmm import TecLinearPhase, TecOnlyOriginal
from born_rime.optimize import minimize
from functools import partial
from born_rime.variational_hmm.nlds_smoother import NonLinearDynamicsSmoother
from born_rime.variational_hmm.utils import constrain_std, deconstrain_std
from jax import numpy as jnp
from jax import disable_jit, jit, grad, vmap
from jax.config import config
from jax.test_util import check_grads
from jax import random

config.update("jax_enable_x64", True)

import numpy as onp

def generate_data():
    import numpy as onp
    onp.random.seed(0)
    T = 10
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV  # + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y
    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = 10. ** 2 * jnp.eye(1)
    mu0 = jnp.zeros(1)
    Gamma0 = 100. ** 2 * jnp.eye(1)
    amp = jnp.ones_like(phase)
    return Gamma0, Omega, Sigma, T, Y_obs, amp, mu0, tec, freqs

def test_problem():
    onp.random.seed(0)
    Gamma0, Omega, Sigma, T, Y_obs, amps, mu0, tec, freqs = generate_data()

    sigma = jnp.sqrt(jnp.diag(Sigma))
    gamma0 = jnp.sqrt(jnp.diag(Gamma0))

    m = FailWhenCalledInThis(freqs)
    x0 = jnp.concatenate([mu0, deconstrain_std(gamma0)])
    for y, amp in zip(Y_obs, amps):

        def neg_elbo(params):
            mu, gamma = params
            gamma = constrain_std(gamma)
            res = m.neg_elbo(freqs, y, sigma, amp, mu, gamma, mu0, gamma0)
            return res
        # check_grads(neg_elbo, (x0,), 3)
        print(grad(neg_elbo)(x0))

    f1 = TecOnlyOriginal(freqs)
    f2 = TecLinearPhase(freqs)
    mu = mu0
    gamma = gamma0
    for y, amp in zip(Y_obs, amps):
        g1 = grad(lambda _mu, _gamma: f1.neg_elbo(freqs, y, sigma, amp, _mu[0], _gamma[0], mu0, jnp.sqrt(jnp.diag(Gamma0))), argnums=[0, 1])
        g2 = grad(lambda _mu, _gamma: f2.neg_elbo(freqs, y, sigma, amp, _mu, _gamma, mu0, jnp.sqrt(jnp.diag(Gamma0))), argnums=[0, 1])
        print('original negelbo grad', g1(mu, gamma), 'linearPhase negelbo g', g2(mu, gamma))
        assert jnp.isclose(g1(mu, gamma), g2(mu, gamma)).all()

        # check_grads(lambda _mu, _gamma: f1.neg_elbo(freqs, y, sigma, amp, _mu[0], _gamma[0], mu0, jnp.sqrt(jnp.diag(Gamma0))), (mu, gamma), 3)
        # check_grads(lambda _mu, _gamma: f2.neg_elbo(freqs, y, sigma, amp, _mu, _gamma, mu0, jnp.sqrt(jnp.diag(Gamma0))), (mu, gamma), 3)

        # def negelbo(params):
        #     mu, gamma = params[0], constrain_std(params[1])
        #     return f2.neg_elbo(freqs, y, sigma, amp, mu, gamma, mu0, jnp.sqrt(jnp.diag(Gamma0)))
        #
        # result = minimize(negelbo, jnp.zeros(2),method='bfgs')
        # print(result)


class FailWhenCalledInThis(object):
    def __init__(self, freqs):
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    @property
    def num_control_params(self):
        return 1

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        phase = mu[0] * self.tec_conv
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))

        def neg_elbo(params):
            mu, gamma = params
            gamma = constrain_std(gamma)
            # print("mu {}\n gamma {}\n Y {}\n sigma {}\n amp {}\n prior_mu {}\n prior_gamma {}".format(mu, gamma, Y, sigma, amp, prior_mu, prior_gamma))
            res = self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)
            print(Y)
            return res

        @jit
        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, gtol=1e-6))
            return result.x

        x0 = jnp.concatenate([prior_mu, deconstrain_std(prior_gamma)])
        from jax.test_util import check_grads
        check_grads(neg_elbo, (x0,),3)
        print('f and Jac',value_and_jacobian(neg_elbo)(x0))
        x1 = do_minimisation(x0)
        # basin = jnp.mean(jnp.abs(jnp.pi / self.tec_conv)) * 0.5
        # num_basin = int(self.tec_scale / basin) + 1
        #
        # obj_try = jnp.stack(
        #     [neg_elbo(jnp.array([x1[0] + i * basin, x1[1]])) for i in range(-num_basin, num_basin + 1, 1)],
        #     axis=0)
        # which_basin = jnp.argmin(obj_try, axis=0)
        # x0_next = jnp.array([x1[0] + (which_basin - float(num_basin)) * basin, x1[1]])
        # x2 = do_minimisation(x0_next)
        x2 = x1

        tec_mean = x2[0]
        tec_uncert = constrain_std(x2[1])

        post_mu = jnp.array([tec_mean])
        post_cov = jnp.array([[tec_uncert ** 2]])

        return post_mu, post_cov

    def neg_elbo(self, freqs, Y_obs, sigma, amp, mu, gamma, mu_prior, gamma_prior):
        return scalar_KL(mu, gamma, mu_prior, gamma_prior) - self.var_exp(freqs, Y_obs, sigma, amp, mu, gamma)

    def var_exp(self, freqs, Y_obs, sigma, amp, mu, gamma):
        """
        Computes variational expectation
        Args:
            freqs: [Nf]
            Y_obs: [Nf]
            sigma: [Nf]
            amp: [Nf]
            mu: scalar
            gamma: scalar

        Returns: scalar

        """
        Nf = freqs.size
        sigma_real = sigma[:Nf]
        sigma_imag = sigma[Nf:]
        m = mu
        l = gamma
        amps = amp
        Yreal = Y_obs[:Nf]
        Yimag = Y_obs[Nf:]
        a = 1. / sigma_real
        b = 1. / sigma_imag
        phi = self.tec_conv * m
        theta = self.tec_conv ** 2 * l * l
        res = -b ** 2 * (amps ** 2 + 2. * Yimag ** 2)
        res += -a ** 2 * (amps ** 2 + 2. * Yreal ** 2)
        res += -4. * jnp.log(2. * jnp.pi / (a * b))
        res += amps * jnp.exp(-2. * theta) * (
                amps * (b ** 2 - a ** 2) * jnp.cos(2. * phi) + 4. * jnp.exp(1.5 * theta) * (
                a ** 2 * Yreal * jnp.cos(phi) + b ** 2 * Yimag * jnp.sin(phi)))
        res *= 0.25
        return jnp.sum(res, axis=-1)