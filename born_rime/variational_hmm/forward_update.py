import born_rime.variational_hmm
from born_rime.optimize import minimize
from born_rime.variational_hmm.utils import constrain_std, deconstrain_std, constrain_tec, deconstrain_tec, \
    constrain_omega, deconstrain_omega, \
    constrain_sigma, deconstrain_sigma, scalar_KL
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
        phase = mu[0] * self.tec_conv
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))

        def neg_elbo(params):
            mu, gamma = params
            gamma = constrain_std(gamma)
            return self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)

        @jit
        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, g_tol=1e-6))
            return result.x

        x0 = jnp.concatenate([prior_mu, deconstrain_std(prior_gamma)])
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


class AmpDiagLinearPhaseDiagSigma(ForwardUpdateEquation):
    def __init__(self, freqs):
        self.freqs = freqs

    @property
    def num_control_params(self):
        return 1

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        raise NotImplementedError()

    @property
    def _phase_basis_size(self):
        raise NotImplementedError()

    def forward_model(self, mu, *control_params):
        """
        Return the model data.
        Args:
            mu: [K]

        Returns:
            Model data [N]

        """
        amp = control_params[0]
        f = self._phase_basis(self.freqs)  # Nf,M
        phase = jnp.dot(f, mu)  # Nf
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=0)

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, *control_params):
        amp = control_params[0]

        sigma = jnp.sqrt(jnp.diag(Sigma))
        prior_gamma = jnp.sqrt(jnp.diag(prior_Gamma))

        def neg_elbo(params):
            mu = params[:self._phase_basis_size]
            gamma = constrain_std(params[self._phase_basis_size:])
            return self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)

        @jit
        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, g_tol=1e-6))
            return result.x

        x0 = jnp.concatenate([prior_mu, deconstrain_std(prior_gamma)])
        x1 = do_minimisation(x0)

        post_mu = x1[:self._phase_basis_size]
        post_gamma = constrain_std(x1[self._phase_basis_size:])
        post_Gamma = jnp.diag(jnp.square(post_gamma))

        return post_mu, post_Gamma

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
            mu: [M]
            gamma: [M]

        Returns: scalar

        """
        f = self._phase_basis(self.freqs)  # Nf,M
        Nf = freqs.size
        Sigma_real = jnp.square(sigma[:Nf])
        Sigma_imag = jnp.square(sigma[Nf:])
        Yreal = Y_obs[:Nf]
        Yimag = Y_obs[Nf:]

        phi = jnp.dot(f, mu)
        theta = 0.5 * jnp.dot(jnp.square(f), jnp.square(gamma))

        var_exp = -2. * Nf * jnp.log(2. * jnp.pi)
        var_exp -= jnp.sum(jnp.log(sigma))
        var_exp -= (Yreal * (Yreal - 2. * amp * jnp.exp(-theta) * jnp.cos(phi))
                    + 0.5 * jnp.square(amp) * (1. + jnp.exp(-2. * theta) * jnp.cos(2. * phi))) / Sigma_real
        var_exp -= (Yimag * (Yimag - 2. * amp * jnp.exp(-theta) * jnp.sin(phi))
                    + 0.5 * jnp.square(amp) * (1. - jnp.exp(-2. * theta) * jnp.cos(2. * phi))) / Sigma_imag

        return 0.5 * jnp.sum(var_exp)

class TecOnlyAmpDiagLinearPhaseDiagSigma(AmpDiagLinearPhaseDiagSigma):
    def __init__(self, freqs):
        super(TecOnlyAmpDiagLinearPhaseDiagSigma, self).__init__(freqs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return self.tec_conv[:, None]

    @property
    def _phase_basis_size(self):
        return 1

class TecClockAmpDiagLinearPhaseDiagSigma(AmpDiagLinearPhaseDiagSigma):
    def __init__(self, freqs):
        super(TecClockAmpDiagLinearPhaseDiagSigma, self).__init__(freqs)
        self.freqs = freqs
        self.tec_conv = -8.4479745e6 / freqs

    def _phase_basis(self, freqs):
        """
        Returns the linease phase basis as a function of freq.
        Args:
            freqs: [Nf] frequency

        Returns:
            [Nf, M] basis
        """
        return jnp.concatenate([self.tec_conv[:, None], jnp.ones((freqs.shape[0], 1))/100.], axis=1)

    @property
    def _phase_basis_size(self):
        return 2