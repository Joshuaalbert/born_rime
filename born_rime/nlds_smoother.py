import jax.numpy as np
from jax.lax import scan, while_loop
from jax import jit, tree_multimap
from .optimize import minimize
from collections import namedtuple
from functools import partial
from jax.config import config

config.update("jax_enable_x64", True)


def pyscan(f, init, xs, reverse=False):
    length = xs[0].shape[0]
    carry = init
    ys = []
    for i in range(length):
        if reverse:
            i = length - i - 1
        x = tree_multimap(lambda v: v[i, ...], xs)
        carry, y = f(carry, x)
        ys.append(y)
    keys = ys[0]._asdict().keys()
    if not reverse:
        d = {k: np.stack([y._asdict()[k] for y in ys], axis=0) for k in keys}
    else:
        d = {k: np.stack([y._asdict()[k] for y in ys[::-1]], axis=0) for k in keys}
    ys = ys[0]._replace(**d)
    return carry, ys


class ForwardUpdateEquation(object):
    def neg_elbo(self, *args):
        raise NotImplementedError()

    def E_update(self, n, prior_mu, prior_Gamma, Y, Sigma, **control_params):
        return prior_mu, prior_Gamma

    def M_update(self, post_mu_n1, post_Gamma_n1, post_mu_n, post_Gamma_n,
                 Y, Sigma_n1, Omega_n1, **control_params):
        raise NotImplementedError()


def constrain(v, a, b):
    return a + (np.tanh(v) + 1) * (b - a) / 2.


def deconstrain(v, a, b):
    return np.arctanh(np.clip((v - a) * 2. / (b - a) - 1., -0.999, 0.999))


def constrain_tec(v, lower=0.1, scale=1.):
    return scale * np.log(np.exp(v) + 1.) + lower


def deconstrain_tec(v, lower=0.1, scale=1.):
    y = np.maximum(np.exp((v - lower) / scale) - 1., 0.)
    return np.maximum(-1e3, np.log(y))


def constrain_omega(v, lower=0.5, scale=10.):
    return scale * np.log(np.exp(v) + 1.) + lower


def deconstrain_omega(v, lower=0.5, scale=10.):
    y = np.maximum(np.exp((v - lower) / scale) - 1., 0.)
    return np.maximum(-1e3, np.log(y))


def constrain_sigma(v, lower=0.01, scale=0.5):
    return scale * np.log(np.exp(v) + 1.) + lower


def deconstrain_sigma(v, lower=0.01, scale=0.5):
    y = np.maximum(np.exp((v - lower) / scale) - 1., 0.)
    return np.maximum(-1e3, np.log(y))


def test_constrain():
    assert np.isclose(0.5, constrain_sigma(deconstrain_sigma(0.5)))
    assert np.isclose(0.01, constrain_sigma(deconstrain_sigma(0.01)))
    assert np.isclose(0.01, constrain_sigma(deconstrain_sigma(0.001)))


def test_E_step():
    import numpy as onp
    onp.random.seed(1)
    import jax
    T = 1
    tec = np.array([50.])  # np.cumsum(10.*onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = np.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
    Y_obs = Y + 1. * onp.random.normal(size=Y.shape)

    update = TecAmpsDiagSigmaDiagOmega(freqs)

    Sigma = 1. ** 2 * np.eye(48)
    Omega = 10. * np.eye(1)
    mu0 = np.zeros(1)
    Gamma0 = 20. ** 2 * np.eye(1)
    amp = np.ones_like(phase)
    E_update = jax.jit(update.E_update)
    import timeit
    import numpy as onp
    print(timeit.timeit(
        lambda: onp.array(E_update(onp.random.normal(size=1), Gamma0, Y_obs[0, ...], Sigma, amp=amp[0, ...])[0]),
        number=3))
    print(E_update(mu0, Gamma0, Y_obs[0, ...], Sigma, amp=amp[0, ...]))
    print(tec)

    # Sigma = np.broadcast_to(Sigma, Y.shape[0:1] + Sigma.shape[-2:])
    # Omega = np.broadcast_to(Omega, Y.shape[0:1] + Omega.shape[-2:])
    # # print(jax.make_jaxpr(hmm)(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # # print(jax.make_jaxpr(hmm)(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # print(hmm.forward_filter(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # print(tec)
    # res = hmm(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp)
    # print(res)


def test_M_step():
    import numpy as onp
    onp.random.seed(1)
    import jax
    tec = np.array([50., 60.])
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = np.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
    Y_obs = Y + 2. * onp.random.normal(size=Y.shape)

    update = TecAmpsDiagSigmaDiagOmega(freqs)

    Sigma = 2. ** 2 * np.eye(48)
    Omega = 1. ** 2 * np.eye(1)
    amp = np.ones_like(phase)

    post_mu_n1 = np.array([10.])
    post_Gamma_n1 = 3. ** 2 * np.eye(1)

    post_mu_n = np.array([55.])
    post_Gamma_n = 3. ** 2 * np.eye(1)

    res = update.M_update(post_mu_n1, post_Gamma_n1, post_mu_n, post_Gamma_n, Y_obs[1, ...], Sigma, Omega,
                          amp=amp[0, ...])

    print("omega", np.sqrt(np.diag(res[0])), 'sigma', np.sqrt(np.diag(res[1])))


class TecAmpsDiagSigmaDiagOmega(ForwardUpdateEquation):
    def __init__(self, freqs, _nojit=False):
        self.freqs = freqs
        self.tec_scale = 300.
        self.tec_conv = -8.4479745e6 / freqs
        self._nojit = _nojit

    def E_update(self, prior_mu, prior_Gamma, Y, Sigma, **control_params):
        amp = control_params.get('amp')

        sigma = np.sqrt(np.diag(Sigma))
        prior_gamma = np.sqrt(np.diag(prior_Gamma))

        # @jit
        def neg_elbo(params):
            mu, log_gamma = params
            gamma = constrain_tec(log_gamma)
            return self.neg_elbo(self.freqs, Y, sigma, amp, mu, gamma, prior_mu, prior_gamma)

        # @jit
        def do_minimisation(x0):
            result = minimize(neg_elbo, x0, method='BFGS',
                              options=dict(ls_maxiter=100, analytic_initial_hessian=
                              False, g_tol=1e-6), _nojit=self._nojit)
            return result.x
            # result = minimize(neg_elbo, x0, method='BFGS')
            # print(result)
            # #
            # return result.x

        x0 = np.concatenate([prior_mu, deconstrain_tec(np.array([5.]))])
        x1 = do_minimisation(x0)

        basin = np.mean(np.abs(np.pi / self.tec_conv)) * 0.5
        num_basin = int(self.tec_scale / basin) + 1

        obj_try = np.stack(
            [neg_elbo(np.array([x1[0] + i * basin, x1[1]])) for i in range(-num_basin, num_basin + 1, 1)],
            axis=0)
        which_basin = np.argmin(obj_try, axis=0)
        x0_next = np.array([x1[0] + (which_basin - float(num_basin)) * basin, x1[1]])
        x2 = do_minimisation(x0_next)

        tec_mean = x2[0]
        tec_uncert = constrain_tec(x2[1])

        post_mu = np.array([tec_mean], np.float64)
        post_cov = np.array([[tec_uncert ** 2]], np.float64)

        return post_mu, post_cov

    def M_update(self, post_mu_n1, post_Gamma_n1, post_mu_n, post_Gamma_n,
                 Y, Sigma_n1, Omega_n1, **control_params):
        amp = control_params.get('amp')

        log_sigma0 = deconstrain_sigma(np.sqrt(np.diag(Sigma_n1)))
        log_omega0 = deconstrain_omega(np.sqrt(np.diag(Omega_n1)))

        post_gamma_n = np.sqrt(np.diag(post_Gamma_n))

        def neg_elbo(params):
            omega = constrain_omega(params[0:1])
            sigma = constrain_sigma(params[1:])
            prior_gamma_n = np.sqrt(np.diag(post_Gamma_n1) + omega ** 2)
            prior_mu_n = post_mu_n1
            ret = self.neg_elbo(self.freqs, Y, sigma, amp, post_mu_n, post_gamma_n, prior_mu_n, prior_gamma_n)
            return ret

        def do_minimisation(x0):
            return minimize(neg_elbo, x0, method='BFGS',
                            options=dict(ls_maxiter=100, maxiter=5, analytic_initial_hessian=False, g_tol=1e-3),
                            _nojit=self._nojit).x

        x0 = np.concatenate([log_omega0, log_sigma0])
        # x1 = do_minimisation(x0)
        x1 = x0

        omega = constrain_omega(x1[0:1])
        sigma = constrain_sigma(x1[1:])

        Omega = np.diag(omega ** 2)
        Sigma = np.diag(sigma ** 2)

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
        q_var = np.square(gamma)
        var_prior = np.square(prior_gamma)
        trace = q_var / var_prior
        mahalanobis = np.square(mu - prior_mu) / var_prior
        constant = -1.
        logdet_qcov = np.log(var_prior) - np.log(q_var)
        twoKL = mahalanobis + constant + logdet_qcov + trace
        prior_KL = 0.5 * twoKL
        return np.sum(prior_KL)

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
        res += -4. * np.log(2. * np.pi / (a * b))
        res += amps * np.exp(-2. * theta) * (
                amps * (b ** 2 - a ** 2) * np.cos(2. * phi) + 4. * np.exp(1.5 * theta) * (
                a ** 2 * Yreal * np.cos(phi) + b ** 2 * Yimag * np.sin(phi)))
        res *= 0.25
        return np.sum(res, axis=-1)


class NonLinearDynamicsSmoother(object):
    def __init__(self, forward_update_equation: ForwardUpdateEquation, N, diagonal_Sigma=True, diagonal_Omega=True,
                 _nojit=False):
        self.forward_update_equation = forward_update_equation
        self.N = N
        self._nojit = _nojit
        if not (forward_update_equation._nojit == self._nojit):
            raise ValueError("Trying to run with _nojit different for forward equation")

        if not diagonal_Sigma:
            raise ValueError("Can't yet handle full obs. covariances")
        if not diagonal_Omega:
            raise ValueError("Can't yet handle full param covariances")

        self.diagonal_Sigma = diagonal_Sigma
        self.diagonal_Omega = diagonal_Omega

    def __call__(self, Y, Sigma, mu0, Gamma0, Omega, **control_params):
        NonLinearDynamicsSmootherState = namedtuple('NonLinearDynamicsSmootherState',
                                                    ['i', 'Sigma_i1', 'post_mu', 'post_Gamma', 'Omega_i1', 'mu0_i1',
                                                     'Gamma0_i1'])
        T = Y.shape[0]

        Sigma = np.broadcast_to(Sigma, Y.shape[0:1] + Sigma.shape[-2:])
        Omega = np.broadcast_to(Omega, Y.shape[0:1] + Omega.shape[-2:])

        state = NonLinearDynamicsSmootherState(i=0,
                                               post_mu=np.tile(mu0[None, :], [T, 1]),
                                               post_Gamma=np.tile(Gamma0[None, :, :], [T, 1, 1]),
                                               Sigma_i1=Sigma,
                                               Omega_i1=Omega,
                                               mu0_i1=mu0,
                                               Gamma0_i1=Gamma0)

        def body(state):
            # TODO: check b[-1] == f[-1]
            prior_Gamma, post_mu_f, post_Gamma_f = self.forward_filter(Y, state.Sigma_i1, state.mu0_i1, state.Gamma0_i1,
                                                                       state.Omega_i1, **control_params)

            post_mu_b, post_Gamma_b = self.backward_filter(prior_Gamma, post_mu_f, post_Gamma_f)

            mu0_i, Gamma0_i, (Sigma_i, Omega_i) = self.parameter_estimation(post_mu_b, post_Gamma_b, Y, state.Sigma_i1,
                                                                            state.Omega_i1, **control_params)

            state = state._replace(i=state.i + 1, post_mu=post_mu_b, post_Gamma=post_Gamma_b,
                                   Sigma_i1=Sigma_i, Omega_i1=Omega_i, mu0_i1=mu0_i, Gamma0_i1=Gamma0_i)
            return state

        if self._nojit:
            while state.i < self.N:
                state = body(state)
        else:
            state = while_loop(lambda state: state.i < self.N,
                               body,
                               state)

        return state

    # @partial(jit, static_argnums=[0])
    def forward_filter(self, Y, Sigma, mu0, Gamma0, Omega, **control_params):
        ForwardFilterState = namedtuple('ForwardFilterState', ['n', 'post_mu_n1', 'post_Gamma_n1'])
        ForwardFilterResult = namedtuple('ForwardFilterResult', ['prior_Gamma', 'post_mu', 'post_Gamma'])
        ForwardFilterX = namedtuple('ForwardFilterX', ['Y', 'Sigma', 'Omega'] + list(control_params.keys()))

        state = ForwardFilterState(n=0,
                                   post_mu_n1=mu0,
                                   post_Gamma_n1=Gamma0)
        control_params.update(dict(Y=Y, Sigma=Sigma, Omega=Omega))
        X = ForwardFilterX(**control_params)

        def f(state, X):
            prior_mu_n = state.post_mu_n1
            prior_Gamma_n = state.post_Gamma_n1 + X.Omega

            post_mu_n, post_Gamma_n = self.forward_update_equation.E_update(
                prior_mu_n,
                prior_Gamma_n,
                **X._asdict())

            state = state._replace(n=state.n + 1,
                                   post_mu_n1=post_mu_n,
                                   post_Gamma_n1=post_Gamma_n)

            result = ForwardFilterResult(prior_Gamma=prior_Gamma_n, post_mu=post_mu_n, post_Gamma=post_Gamma_n)
            return state, result

        if self._nojit:
            final_state, results = pyscan(f, state, X)
        else:
            final_state, results = scan(f, state, X)

        return results

    # @partial(jit, static_argnums=[0])
    def backward_filter(self, prior_Gamma, post_mu_f, post_Gamma_f):
        BackwardFilterState = namedtuple('BackwardFilterState', ['n1', 'post_mu_n', 'post_Gamma_n'])
        BackwardFilterResult = namedtuple('BackwardFilterResult', ['post_mu', 'post_Gamma'])
        BackwardFilterX = namedtuple('BackwardFilterX', ['post_mu_f_n1', 'post_Gamma_f_n1', 'prior_Gamma_n'])

        X = BackwardFilterX(post_mu_f_n1=post_mu_f[:-1, ...],
                            post_Gamma_f_n1=post_Gamma_f[:-1, ...],
                            prior_Gamma_n=prior_Gamma[1:, ...])

        state = BackwardFilterState(n1=post_mu_f.shape[0] - 2,
                                    post_mu_n=post_mu_f[-1, ...],
                                    post_Gamma_n=post_Gamma_f[-1, ...])

        def f(state, X):
            JT_t1 = np.linalg.solve(X.prior_Gamma_n, X.post_Gamma_f_n1)
            post_mu_n1 = X.post_mu_f_n1 + np.dot(JT_t1.T, (state.post_mu_n - X.post_mu_f_n1)[:, None])[:, 0]
            post_Gamma_n1 = X.post_Gamma_f_n1 + np.dot(JT_t1.T, np.dot(state.post_Gamma_n - X.prior_Gamma_n, JT_t1))

            state = state._replace(n1=state.n1 - 1,
                                   post_mu_n=post_mu_n1,
                                   post_Gamma_n=post_Gamma_n1)

            result = BackwardFilterResult(post_mu=post_mu_n1,
                                          post_Gamma=post_Gamma_n1)
            return state, result

        if self._nojit:
            final_state, results = pyscan(f, state, X, reverse=True)
        else:
            final_state, results = scan(f, state, X, reverse=True)

        results = results._replace(post_mu=np.concatenate([results.post_mu, post_mu_f[-1:, ...]], axis=0),
                                   post_Gamma=np.concatenate([results.post_Gamma, post_Gamma_f[-1:, ...]], axis=0))

        return results

    def parameter_estimation(self, post_mu_b, post_Gamma_b, Y, Sigma, Omega, **control_params):
        """
        Maximise E[log likelihood] over posterior
        Returns:
        """

        MstepState = namedtuple('MstepState', ['n'])
        MstepResult = namedtuple('MstepResult', ['Sigma', 'Omega'])
        MstepX = namedtuple('MstepX',
                            ['Y', 'post_mu_n1', 'post_Gamma_n1', 'post_mu_n', 'post_Gamma_n', 'control_params',
                             'Sigma_n', 'Omega_n'])

        mu0 = post_mu_b[0, ...]
        Gamma0 = post_Gamma_b[0, ...]

        state = MstepState(n=0)

        X = MstepX(Y=Y,
                   post_mu_n1=np.concatenate([post_mu_b[0:1, ...], post_mu_b[:-1, ...]], axis=0),
                   post_Gamma_n1=np.concatenate([post_Gamma_b[0:1, ...], post_Gamma_b[:-1, ...]], axis=0),
                   post_mu_n=post_mu_b,
                   post_Gamma_n=post_Gamma_b,
                   control_params=control_params,
                   Sigma_n=Sigma,
                   Omega_n=Omega)

        def f(state, X):
            Omega_n, Sigma_n = self.forward_update_equation.M_update(post_mu_n1=X.post_mu_n1,
                                                                     post_Gamma_n1=X.post_Gamma_n1,
                                                                     post_mu_n=X.post_mu_n,
                                                                     post_Gamma_n=X.post_Gamma_n,
                                                                     Y=X.Y,
                                                                     Sigma_n1=X.Sigma_n,
                                                                     Omega_n1=X.Omega_n,
                                                                     **X.control_params)
            state = state._replace(n=state.n+1)
            result = MstepResult(Sigma=Sigma_n, Omega=Omega_n)
            return state, result

        if self._nojit:
            final_state, results = pyscan(f, state, X)
        else:
            final_state, results = scan(f, state, X)

        return mu0, Gamma0, results


def test_nlds_smoother():
    import numpy as onp
    onp.random.seed(0)
    import jax
    T = 128
    tec = np.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = np.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV
    Y = np.concatenate([np.cos(phase), np.sin(phase)], axis=1)
    Y_obs = Y + 0.9 * onp.random.normal(size=Y.shape)
    hmm = NonLinearDynamicsSmoother(TecAmpsDiagSigmaDiagOmega(freqs, _nojit=False), 1, _nojit=False)

    Sigma = 0.9 ** 2 * np.eye(48)
    Omega = 10. * np.eye(1)
    mu0 = np.zeros(1)
    Gamma0 = 100. * np.eye(1)
    amp = np.ones_like(phase)
    # Sigma = np.broadcast_to(Sigma, Y.shape[0:1] + Sigma.shape[-2:])
    # Omega = np.broadcast_to(Omega, Y.shape[0:1] + Omega.shape[-2:])

    # print(jax.make_jaxpr(hmm)(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # print(jax.make_jaxpr(hmm)(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # print(hmm.forward_filter(Y_obs, Sigma, mu0, Gamma0, Omega,amp=amp))
    # print(tec)
    res = jit(hmm)(Y_obs, Sigma, mu0, Gamma0, Omega, amp=amp)
    print(res)
    assert res.post_mu.shape[0] == T
    print(tec - res.post_mu[:,0])