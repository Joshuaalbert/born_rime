import born_rime.variational_hmm
from born_rime.optimize import minimize
from born_rime.variational_hmm.utils import constrain_tec, deconstrain_tec, constrain_omega, deconstrain_omega, \
    constrain_sigma, deconstrain_sigma
from jax import numpy as jnp
from jax import jit


class ForwardUpdateEquation(object):

    @property
    def num_control_params(self):
        """
        Number of control parameters expected.
        Returns: int
        """
        raise NotImplementedError()

    def neg_elbo(self, *args):
        """
        Return the negative ELBO.
        Args:
            *args:

        Returns:

        """
        raise NotImplementedError()

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]
            *control_params: list of any other arrays
        Returns:
            Model data [N]

        """
        raise NotImplementedError()

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        """
        Given the current data and control params as well as a Gaussian prior, return the conditional mean and covariance
        of a Gaussian variational posterior.

        Args:
            prior_mu: [K] prior mean
            prior_Gamma: [K,K] prior covariance
            Y: [N] observed data
            Sigma: [N,N] Observed data covariance
            *control_params: list of arrays of arbitrary shape.

        Returns:
            posterior mean [K]
            posterior covariance [K,K]
        """
        return prior_mu, prior_Gamma

class TecAmpsDiagSigmaDiagOmega(ForwardUpdateEquation):
    def __init__(self, freqs):
        self.freqs = freqs
        self.tec_scale = 300.
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
        phase = mu[0]*self.tec_conv
        return jnp.concatenate([amp*jnp.cos(phase), amp*jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))

        def neg_elbo(params):
            mu, gamma = params
            gamma = constrain_tec(gamma)
            return self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)

        @jit
        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, g_tol=1e-6))
            return result.x

        x0 = jnp.concatenate([prior_mu, deconstrain_tec(jnp.array([5.]))])
        x1 = do_minimisation(x0)

        basin = jnp.mean(jnp.abs(jnp.pi / self.tec_conv)) * 0.5
        num_basin = int(self.tec_scale / basin) + 1

        obj_try = jnp.stack(
            [neg_elbo(jnp.array([x1[0] + i * basin, x1[1]])) for i in range(-num_basin, num_basin + 1, 1)],
            axis=0)
        which_basin = jnp.argmin(obj_try, axis=0)
        x0_next = jnp.array([x1[0] + (which_basin - float(num_basin)) * basin, x1[1]])
        x2 = do_minimisation(x0_next)

        tec_mean = x2[0]
        tec_uncert = constrain_tec(x2[1])

        post_mu = jnp.array([tec_mean], jnp.float64)
        post_cov = jnp.array([[tec_uncert ** 2]], jnp.float64)

        return post_mu, post_cov

    def _M_update(self, post_mu_n1, post_Gamma_n1, post_mu_n, post_Gamma_n,
                 Y, Sigma_n1, Omega_n1, **control_params):
        amp = control_params.get('amp')

        log_sigma0 = deconstrain_sigma(jnp.sqrt(jnp.diag(Sigma_n1)))
        log_omega0 = deconstrain_omega(jnp.sqrt(jnp.diag(Omega_n1)))

        post_gamma_n = jnp.sqrt(jnp.diag(post_Gamma_n))

        def neg_elbo(params):
            omega = constrain_omega(params[0:1])
            sigma = constrain_sigma(params[1:])
            prior_gamma_n = jnp.sqrt(jnp.diag(post_Gamma_n1) + omega ** 2)
            prior_mu_n = post_mu_n1
            ret = self.neg_elbo(self.freqs, Y, sigma, amp, post_mu_n, post_gamma_n, prior_mu_n, prior_gamma_n)
            return ret

        def do_minimisation(x0):
            return minimize(neg_elbo, x0, method='BFGS',
                            options=dict(ls_maxiter=100, maxiter=5, g_tol=1e-3)).x

        x0 = jnp.concatenate([log_omega0, log_sigma0])
        # x1 = do_minimisation(x0)
        x1 = x0

        omega = constrain_omega(x1[0:1])
        sigma = constrain_sigma(x1[1:])

        Omega = jnp.diag(omega ** 2)
        Sigma = jnp.diag(sigma ** 2)

        return Omega, Sigma

    def neg_elbo(self, freqs, Y_obs, sigma, amp, mu, gamma, mu_prior, gamma_prior):
        return self.scalar_KL(mu, gamma, mu_prior, gamma_prior) - self.var_exp(freqs, Y_obs, sigma, amp, mu, gamma)

    def scalar_KL(self, mu, gamma, prior_mu, prior_gamma):
        """
        mean, uncert : [M]
        mean_prior,uncert_prior: [M]
        :return: scalar
        """
        # Get KL
        q_var = jnp.square(gamma)
        var_prior = jnp.square(prior_gamma)
        trace = q_var / var_prior
        mahalanobis = jnp.square(mu - prior_mu) / var_prior
        constant = -1.
        logdet_qcov = jnp.log(var_prior) - jnp.log(q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        prior_KL = 0.5 * twoKL
        return jnp.sum(prior_KL)

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
        TEC_CONV = -8.4479745e6  # mTECU/Hz
        Nf = freqs.size
        sigma_real = sigma[:Nf]
        sigma_imag = sigma[Nf:]
        tec_conv = TEC_CONV / freqs
        m = mu
        l = gamma
        amps = amp
        Yreal = Y_obs[:Nf]
        Yimag = Y_obs[Nf:]
        a = 1. / sigma_real
        b = 1. / sigma_imag
        phi = tec_conv * m
        theta = tec_conv ** 2 * l * l
        res = -b ** 2 * (amps ** 2 + 2. * Yimag ** 2)
        res += -a ** 2 * (amps ** 2 + 2. * Yreal ** 2)
        res += -4. * jnp.log(2. * jnp.pi / (a * b))
        res += amps * jnp.exp(-2. * theta) * (
                amps * (b ** 2 - a ** 2) * jnp.cos(2. * phi) + 4. * jnp.exp(1.5 * theta) * (
                a ** 2 * Yreal * jnp.cos(phi) + b ** 2 * Yimag * jnp.sin(phi)))
        res *= 0.25
        return jnp.sum(res, axis=-1)