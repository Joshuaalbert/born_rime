
import born_rime.variational_hmm
from born_rime.variational_hmm.utils import scalar_KL, mvn_kl, fill_triangular


import jax.numpy as np
import jax
from jax import jit
import numpy as onp
from born_rime import TEC_CONV
from functools import partial


class LinearPhaseModel(object):
    def __init__(self, diagonal=True, do_phase=True, do_amp=False, diagonal_params=True):
        self.diagonal = diagonal
        self.do_phase = do_phase
        self.do_amp = do_amp
        self.diagonal_params = diagonal_params

    @property
    def M_phase(self):
        if not self.do_phase:
            return 1
        return self._M_phase()

    def _M_phase(self):
        raise NotImplementedError()

    @property
    def M_amp(self):
        if not self.do_amp:
            return 1
        return self._M_amp()

    def _M_amp(self):
        raise NotImplementedError()

    def phase_bases(self, freqs):
        """
        Return phase bases matrix of shape [Nfreqs, M]
        :param freqs: [Nfreqs]
        :return:  [Nfreqs, M ]
        """
        if not self.do_phase:
            return np.zeros_like(freqs)[None, :]
        return self._phase_bases(freqs)

    def _phase_bases(self, freqs):
        raise NotImplementedError()

    def amp_bases(self, freqs):
        """
        Return amp bases matrix of shape [Nfreqs, M ]
        :param freqs: [Nfreqs]
        :return:  [Nfreqs, M ]
        """
        if not self.do_amp:
            return np.zeros_like(freqs)[None, :]
        return self._amp_bases(freqs)

    def _amp_bases(self, freqs):
        raise NotImplementedError()

    def _parse_flat_params(self, x):
        params = {}
        idx = 0
        if self.do_phase:
            mu_phase = x[idx:idx + self.M_phase]
            params['mu_phase'] = mu_phase
            idx += self.M_phase
            if self.diagonal_params:
                log_uncert_phase = x[idx:idx + self.M_phase]
                params['uncert_phase'] = np.exp(log_uncert_phase)
                params['Cov_phase'] = np.diag(np.square(params['uncert_phase']))
                idx += self.M_phase
            else:
                m = self.M_phase * (self.M_phase + 1) // 2
                L_phase = fill_triangular(x[idx:idx + m])
                params['L_phase'] = L_phase
                params['Cov_phase'] = np.dot(L_phase, L_phase.T)
                idx += m
        else:
            params['mu_phase'] = np.zeros(self.M_phase)
            if self.diagonal_params:
                params['uncert_phase'] = np.zeros(self.M_phase)
                params['Cov_phase'] = np.diag(np.square(params['uncert_phase']))
            else:
                m = self.M_phase * (self.M_phase + 1) // 2
                L_phase = np.zeros((m, m))
                params['L_phase'] = L_phase
                params['Cov_phase'] = np.dot(L_phase, L_phase.T)

        if self.do_amp:
            mu_amp = x[idx:idx + self.M_amp]
            params['mu_amp'] = mu_amp
            idx += self.M_amp
            if self.diagonal_params:
                log_uncert_amp = x[idx:idx + self.M_amp]
                params['uncert_amp'] = np.exp(log_uncert_amp)
                params['Cov_amp'] = np.diag(np.square(params['uncert_amp']))
                idx += self.M_amp
            else:
                m = self.M_amp * (self.M_amp + 1) // 2
                L_amp = fill_triangular(x[idx:idx + m])
                params['L_amp'] = L_amp
                params['Cov_amp'] = np.dot(L_amp, L_amp.T)
                idx += m
        else:
            params['mu_amp'] = np.zeros(self.M_amp)
            if self.diagonal_params:
                params['uncert_amp'] = np.zeros(self.M_amp)
                params['Cov_amp'] = np.diag(np.square(params['uncert_amp']))
            else:
                m = self.M_amp * (self.M_amp + 1) // 2
                L_amp = np.zeros((m, m))
                params['L_amp'] = L_amp
                params['Cov_amp'] = np.dot(L_amp, L_amp.T)

        return params

    def build_neg_elbo(self, freqs, Y_obs, Sigma, mean_prior_phase=None, uncert_prior_phase=None, mean_prior_amp=None,
                       uncert_prior_amp=None):
        if self.do_phase and mean_prior_phase is None:
            raise ValueError("Missing mean prior phase")
        if self.do_phase and uncert_prior_phase is None:
            raise ValueError("Missing uncert prior phase")
        if self.do_amp and mean_prior_amp is None:
            raise ValueError("Missing mean prior amp")
        if self.do_amp and uncert_prior_amp is None:
            raise ValueError("Missing uncert prior amp")

        def neg_elbo(x):
            params = self._parse_flat_params(x)
            var_exp = self.var_exp(freqs, Y_obs, Sigma, **params)

            # only do the correct KL
            if self.do_phase and not self.do_amp:
                if self.diagonal_params:
                    KL = scalar_KL(params['mu_phase'],
                                   params['uncert_phase'],
                                   mean_prior_phase,
                                   uncert_prior_phase)
                else:
                    KL = mvn_kl(params['mu_phase'],
                                params['L_phase'],
                                mean_prior_phase,
                                uncert_prior_phase)
            elif self.do_phase and self.do_amp:
                if self.diagonal_params:
                    KL = scalar_KL(np.concatenate([params['mu_phase'], params['mu_amp']]),
                                   np.concatenate([params['uncert_phase'], params['uncert_amp']]),
                                   np.concatenate([mean_prior_phase, mean_prior_amp]),
                                   np.concatenate([uncert_prior_phase, uncert_prior_amp]))
                else:
                    KL = mvn_kl(params['mu_phase'],
                                params['L_phase'],
                                mean_prior_phase,
                                uncert_prior_phase) \
                         + \
                         mvn_kl(params['mu_amp'],
                                params['L_amp'],
                                mean_prior_amp,
                                uncert_prior_amp)
            if not self.do_phase and self.do_amp:
                if self.diagonal_params:
                    KL = scalar_KL(params['mu_amp'],
                                   params['uncert_amp'],
                                   mean_prior_amp,
                                   uncert_prior_amp)
                else:
                    KL = mvn_kl(params['mu_amp'],
                                params['L_amp'],
                                mean_prior_amp,
                                uncert_prior_amp)

            return KL - var_exp

        return neg_elbo

    @partial(jit, static_argnums=(0,))
    def var_exp(self, freqs, Y_obs, Sigma, mu_phase, Cov_phase, mu_amp, Cov_amp, **unused_kwargs):
        """
        Computes the variational expectation of a linear phase model.
        Assume Y(nu_i) = g(nu_i) * Exp[I (sum_m^M f_m(nu_i) alpha_m]
        where alpha_m ~ N[mu_m, sigma_m^2]

        Compute E[log(N[Y_obs | Y, Sigma])] over alpha_m distributions

        :param mu: [M]
        :return:
        """

        Nfreq = freqs.size
        zeros_Nf = np.zeros(Nfreq, dtype=np.complex_)
        # Nf,M
        f = self.phase_bases(freqs)
        F = self.amp_bases(freqs)

        def _log_expectation(u, mu, Cov):
            """
            :param u: [..., M]
            :param mu: [M]
            :param Cov: [M,M]
            :return: [...]
            """
            return np.dot(u, mu) + 0.5 * np.sum(np.dot(u, Cov) * u, axis=-1)

        # Nfreqs
        Yreal = np.real(Y_obs)
        Yimag = np.imag(Y_obs)

        if self.diagonal:

            Sigma_inv = np.reciprocal(np.diag(Sigma))

            if self.do_amp:
                Tm2F = _log_expectation(-2. * F, mu_amp, Cov_amp)
                TmF = _log_expectation(-F, mu_amp, Cov_amp)
            else:
                Tm2F = zeros_Nf
                TmF = zeros_Nf

            if self.do_phase:
                Tm2if = _log_expectation(-2j * f, mu_phase, Cov_phase)
                T2if = _log_expectation(2j * f, mu_phase, Cov_phase)
                Tif = _log_expectation(1j * f, mu_phase, Cov_phase)
                Tmif = _log_expectation(-1j * f, mu_phase, Cov_phase)
            else:
                Tm2if = zeros_Nf
                T2if = zeros_Nf
                Tif = zeros_Nf
                Tmif = zeros_Nf

            I_rr = 0.25 * np.sum([
                2. * np.exp(Tm2F),
                np.exp(Tm2if + Tm2F),
                np.exp(T2if + Tm2F),
                -4. * Yreal * np.exp(Tmif + TmF),
                -4. * Yreal * np.exp(Tif + TmF),
                4. * Yreal ** 2
            ], axis=0)

            I_ii = 0.25 * np.sum([
                2. * np.exp(Tm2F),
                -np.exp(Tm2if + Tm2F),
                -np.exp(T2if + Tm2F),
                -4j * Yimag * np.exp(Tmif + TmF),
                +4j * Yimag * np.exp(Tif + TmF),
                4. * Yimag ** 2
            ], axis=0)

            A = -Nfreq * np.log(2. * np.pi) - 0.5 * np.sum(np.log(np.diag(Sigma)))
            B = -0.5 * np.real(np.sum(Sigma_inv[:Nfreq] * I_rr) + np.sum(Sigma_inv[Nfreq:] * I_ii))

            return A + B


        else:
            Sigma_inv = np.linalg.inv(Sigma)

            fi = f[:, :, None]
            fj = f[:, None, :]
            Fi = F[:, :, None]
            Fj = F[:, None, :]
            Yreali = Yreal[:, None]
            Yrealj = Yreal[None, :]
            Yimagi = Yimag[:, None]
            Yimagj = Yimag[None, :]

            if self.do_amp:
                TmFimFj = _log_expectation(-Fi - Fj, mu_amp, Cov_amp)
                TmFj = _log_expectation(-Fj, mu_amp, Cov_amp)
                TmFi = _log_expectation(-Fi, mu_amp, Cov_amp)
            else:
                TmFimFj = zeros_Nf
                TmFj = zeros_Nf
                TmFi = zeros_Nf

            if self.do_phase:
                Tmifimfj = _log_expectation(-1j * (fi - fj), mu_phase, Cov_phase)
                Tifimfj = _log_expectation(1j * (fi - fj), mu_phase, Cov_phase)
                Tmifipfj = _log_expectation(-1j * (fi + fj), mu_phase, Cov_phase)
                Tifipfj = _log_expectation(1j * (fi + fj), mu_phase, Cov_phase)
                Tmifj = _log_expectation(-1j * fj, mu_phase, Cov_phase)
                Tifj = _log_expectation(1j * fj, mu_phase, Cov_phase)
                Tmifi = _log_expectation(-1j * fi, mu_phase, Cov_phase)
                Tifi = _log_expectation(1j * fi, mu_phase, Cov_phase)
            else:
                Tmifimfj = zeros_Nf
                Tifimfj = zeros_Nf
                Tmifipfj = zeros_Nf
                Tifipfj = zeros_Nf
                Tmifj = zeros_Nf
                Tifj = zeros_Nf
                Tmifi = zeros_Nf
                Tifi = zeros_Nf

            I_rr = 0.25 * np.sum([np.exp(Tmifimfj + TmFimFj),
                                  np.exp(Tifimfj + TmFimFj),
                                  np.exp(Tmifipfj + TmFimFj),
                                  np.exp(Tifipfj + TmFimFj),
                                  -2. * Yreali * np.exp(Tmifj + TmFj),
                                  -2. * Yreali * np.exp(Tifj + TmFj),
                                  -2. * Yrealj * np.exp(Tmifi + TmFi),
                                  -2. * Yrealj * np.exp(Tifi + TmFi),
                                  4. * Yreali * Yrealj], axis=0)

            I_ii = 0.25 * np.sum([np.exp(Tmifimfj + TmFimFj),
                                  np.exp(Tifimfj + TmFimFj),
                                  -np.exp(Tmifipfj + TmFimFj),
                                  -np.exp(Tifipfj + TmFimFj),
                                  -2j * Yimagi * np.exp(Tmifj + TmFj),
                                  2j * Yimagi * np.exp(Tifj + TmFj),
                                  -2j * Yimagj * np.exp(Tmifi + TmFi),
                                  2j * Yimagj * np.exp(Tifi + TmFi),
                                  4. * Yimagi * Yimagj], axis=0)

            I_ri = 0.25 * np.sum([-1j * np.exp(Tmifimfj + TmFimFj),
                                  1j * np.exp(Tifimfj + TmFimFj),
                                  1j * np.exp(Tmifipfj + TmFimFj),
                                  -1j * np.exp(Tifipfj + TmFimFj),
                                  -2j * Yreali * np.exp(Tmifj + TmFj),
                                  2j * Yreali * np.exp(Tifj + TmFj),
                                  -2. * Yimagj * np.exp(Tmifi + TmFi),
                                  -2. * Yimagj * np.exp(Tifi + TmFi),
                                  4. * Yreali * Yimagj], axis=0)

            A = -Nfreq * np.log(2. * np.pi) - 0.5 * np.log(np.linalg.det(Sigma))
            B = -0.5 * np.real(
                np.sum(Sigma_inv[:Nfreq, :Nfreq] * I_rr) + np.sum(Sigma_inv[Nfreq:, Nfreq:] * I_ii) + 2. * np.sum(
                    Sigma_inv[:Nfreq, Nfreq:] * I_ri))

            return A + B


class TECPhaseModel(LinearPhaseModel):
    def _M_phase(self):
        return 1

    def _M_amp(self):
        return 1

    def _phase_bases(self, freqs):
        return TEC_CONV * np.reciprocal(freqs)[:, None]

    def amp_bases(self, freqs):
        return np.ones_like(freqs)[:, None]



# def minimise(fun, x0):
#     jax.tree_multimap(lambda arr1, arr2: arr1 + arr2, x1, x2)

def test_linear_phase_model_solve():
    onp.random.seed(0)
    freqs = np.linspace(121e6, 166e6, 24)
    tec_true = 87.
    phase_true = tec_true * TEC_CONV / freqs
    Sigma = 0.7 ** 2 * np.eye(freqs.size * 2)
    Y_obs = np.exp(1j * phase_true) + 0.7 * (
            onp.random.normal(size=phase_true.shape) + 1j * onp.random.normal(size=phase_true.shape))
    amp = np.ones(freqs.size)

    model = TECPhaseModel(diagonal=True, do_phase=True, do_amp=False)

    from scipy.optimize import minimize

    mean_prior_phase, uncert_prior_phase = np.array([0.]), np.array([100.])

    x0 = np.array([100., np.log(5.)])
    neg_elbo = model.build_neg_elbo(freqs, Y_obs, Sigma,
                                    mean_prior_phase=mean_prior_phase, uncert_prior_phase=uncert_prior_phase)

    print(neg_elbo(x0))

    value_grad = jit(jax.value_and_grad(neg_elbo, argnums=[0]))

    @jit
    def value_and_jac(x):
        value, jac = value_grad(x)
        return value, np.array(jac[0])

    print(value_and_jac(x0))

    @jit
    def hvp(primals, tangents):
        return jax.jvp(jax.grad(neg_elbo), [primals], [tangents])[1]

    @jit
    def hessian(x):
        return jax.jacfwd(jax.jacrev(neg_elbo))(x)

    import timeit

    print(timeit.timeit(lambda: minimize(value_and_jac,
                                         x0,
                                         method='Newton-CG',
                                         hessp=hvp,
                                         jac=True), number=10))

    print(timeit.timeit(lambda: minimize(value_and_jac,
                                         x0,
                                         method='Newton-CG',
                                         hess=hessian,
                                         jac=True), number=10))

    print(timeit.timeit(lambda: minimize(value_and_jac,
                                         x0,
                                         method='Newton-CG',
                                         jac=True), number=10))

    print(timeit.timeit(lambda: minimize(value_and_jac,
                                         x0,
                                         method='BFGS',
                                         jac=True), number=10))

    print(timeit.timeit(lambda: minimize(value_and_jac,
                                         x0,
                                         jac=True), number=10))

    res = minimize(value_and_jac,
                   x0,
                   method='Newton-CG',
                   hessp=hvp,
                   hess=hessian,
                   jac=True)

    print(res)


def test_linear_phase_model():
    freqs = np.linspace(121e6, 166e6, 24)
    tec_true = 87.
    phase_true = tec_true * TEC_CONV / freqs
    Sigma = 0.2 ** 2 * np.eye(freqs.size * 2)
    Y_obs = np.exp(1j * phase_true) + 0.2 * (
            onp.random.normal(size=phase_true.shape) + 1j * onp.random.normal(size=phase_true.shape))
    amp = np.ones(freqs.size)

    from scipy.stats import multivariate_normal

    def numeric_full_test(freqs, Y_obs, Sigma, mu, sigma):
        param_dist = multivariate_normal(mu, sigma ** 2)
        tec = param_dist.rvs(10000000)
        phase = (TEC_CONV / freqs[None, :]) * tec[:, None]
        Y_mod = np.exp(1j * phase)
        log_prob = multivariate_normal(np.concatenate([Y_obs.real, Y_obs.imag]), Sigma).logpdf(
            np.concatenate([Y_mod.real, Y_mod.imag], axis=1))
        return np.mean(log_prob)

    def test_var_exp_onp(freqs, Y_obs, Sigma, amp, mu, sigma):
        """
        Analytic int log p(y | ddtec, const, sigma^2) N[ddtec | q, Q]  N[const | f, F] dddtec dconst
        :param l:
        :param Yreal:
        :param Yimag:
        :param sigma_real:
        :param sigma_imag:
        :param k:
        :return:
        """
        Nf = freqs.size
        sigma_real = onp.sqrt(onp.diag(Sigma)[:Nf])
        sigma_imag = onp.sqrt(onp.diag(Sigma)[Nf:])
        tec_conv = TEC_CONV / freqs
        m = mu
        l = sigma
        amps = amp
        Yimag = onp.imag(Y_obs)
        Yreal = onp.real(Y_obs)
        a = 1. / sigma_real
        b = 1. / sigma_imag
        phi = tec_conv * m
        theta = tec_conv ** 2 * l * l
        res = -b ** 2 * (amps ** 2 + 2. * Yimag ** 2)
        res += -a ** 2 * (amps ** 2 + 2. * Yreal ** 2)
        res += -4. * onp.log(2. * onp.pi / (a * b))
        res += amps * onp.exp(-2. * theta) * (
                amps * (b ** 2 - a ** 2) * onp.cos(2. * phi) + 4. * onp.exp(1.5 * theta) * (
                a ** 2 * Yreal * onp.cos(phi) + b ** 2 * Yimag * onp.sin(phi)))
        res *= 0.25
        return onp.sum(res, axis=-1)

    # print(model.var_exp(freqs, Y_obs, Sigma, np.array([0.]), np.array([10.])**2, np.array([0.]), np.array([0.])))

    model = TECPhaseModel(diagonal=True, do_amp=False, do_phase=True)

    assert np.isclose(
        model.var_exp(freqs, Y_obs, Sigma, np.array([0.]), np.diag(np.array([10.]) ** 2), np.array([0.]),
                      np.diag(np.array([0.]))),
        test_var_exp_onp(freqs, Y_obs, Sigma, amp, np.array([0.]), np.array([10.]))
    )

    print(
        test_var_exp_onp(freqs, Y_obs, Sigma, amp, np.array([0.]), np.array([10.])),
        numeric_full_test(freqs, Y_obs, Sigma, np.array([0.]), np.array([[10.]]))
    )
    #

    import timeit

    # print(make_jaxpr(model.var_exp)(freqs, Y_obs, Sigma, np.array([0.]), np.diag(np.array([10.]) ** 2), np.array([0.]),
    #                           np.diag(np.array([0.]))))

    print(
        timeit.timeit(lambda: test_var_exp_onp(freqs, Y_obs, Sigma, amp, np.array([0.]), np.array([10.])), number=100))
    print(timeit.timeit(
        lambda: model.var_exp(freqs, Y_obs, Sigma, np.array([0.]), np.diag(np.array([10.]) ** 2), np.array([0.]),
                              np.diag(np.array([0.]))), number=100))

    assert timeit.timeit(
        lambda: model.var_exp(freqs, Y_obs, Sigma, np.array([0.]), np.array([10.]) ** 2, np.array([0.]),
                              np.array([0.])), number=100) < timeit.timeit(
        lambda: test_var_exp_onp(freqs, Y_obs, Sigma, amp, np.array([0.]), np.array([10.])), number=100)

    Sigma = 0.2 * onp.random.normal(size=[freqs.size * 2, freqs.size * 2])
    Sigma = np.dot(Sigma, Sigma.T)

    model = TECPhaseModel(diagonal=False)
    print(model.var_exp(freqs, Y_obs, Sigma, np.array([0.]), np.diag(np.array([10.]) ** 2), np.array([0.]),
                        np.diag(np.array([0.]))))
    print(numeric_full_test(freqs, Y_obs, Sigma, np.array([85.]), np.array([10.])))
