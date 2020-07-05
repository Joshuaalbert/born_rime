from born_rime.variational_hmm import TecClockLinearPhaseLinearised, TecClockLinearPhase
from functools import partial
from born_rime.variational_hmm.nlds_smoother import NonLinearDynamicsSmoother
from jax import numpy as jnp
from jax import jit
from jax.config import config
from typing import NamedTuple

config.update("jax_enable_x64", True)


class SearchState(NamedTuple):
    fe_maxiter: int
    fe_momentum: float
    hmm_maxiter: int
    hmm_momentum: float
    omega_window: int
    sigma_window: int


def debug_param_search():
    T, Y_obs, freqs, onp, phase, plt, tec = generate_data()
    import numpy as onp

    states = []
    fitness = []
    for i in range(100):
        state = SearchState(fe_maxiter=onp.random.randint(1, 2),
                            fe_momentum=onp.random.uniform(),
                            hmm_maxiter=onp.random.randint(1, 2),
                            hmm_momentum=onp.random.uniform(),
                            omega_window=onp.random.randint(1, 50) * 2 + 1,
                            sigma_window=onp.random.randint(1, 50) * 2 + 1)

        res, tec_mae, tec_mse = compute_solution(Y_obs, state.fe_maxiter, state.fe_momentum, freqs, state.hmm_maxiter,
                                        state.hmm_momentum, state.omega_window,
                                        phase, state.sigma_window, tec)

        print(i, state, tec_mae, tec_mse)
        fitness.append(tec_mae)
        states.append(state)


    best = onp.argsort(fitness)
    for i in best[:5]:
        print(i, fitness[i], states[i])




def compute_solution(Y_obs, fe_maxiter, fe_momentum, freqs, hmm_maxiter, hmm_momentum, omega_window, phase,
                     sigma_window, tec):
    hmm = NonLinearDynamicsSmoother(TecClockLinearPhaseLinearised(freqs, tol=[0.5, 0.01],
                                                                  maxiter=fe_maxiter, momentum=fe_momentum))
    Sigma = 0.5 ** 2 * jnp.eye(48)
    Omega = jnp.diag(jnp.array([10., 0.1])) ** 2
    mu0 = jnp.zeros(2)
    Gamma0 = jnp.diag(jnp.array([100., 1.])) ** 2
    amp = jnp.ones_like(phase)
    res = jit(partial(hmm, tol=[0.5, 0.01], maxiter=hmm_maxiter,
                      omega_window=omega_window, sigma_window=sigma_window, momentum=hmm_momentum))(Y_obs, Sigma, mu0,
                                                                                                    Gamma0, Omega, amp)
    tec_mae = jnp.mean(jnp.abs(res.post_mu[:, 0] - tec))
    tec_mse = jnp.sqrt(jnp.mean(jnp.square(res.post_mu[:, 0] - tec)))
    return res, tec_mae, tec_mse


def generate_data():
    import numpy as onp
    import pylab as plt
    onp.random.seed(0)
    T = 1000
    tec = onp.cumsum(10. * onp.random.normal(size=T))
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    freqs = onp.linspace(121e6, 168e6, 24)
    phase = tec[:, None] / freqs * TEC_CONV + onp.linspace(-onp.pi, onp.pi, T)[:, None]
    Y = onp.concatenate([onp.cos(phase), onp.sin(phase)], axis=1)
    Y_obs = Y + 0.25 * onp.random.normal(size=Y.shape)
    Y_obs[500:550:2, :] += 3. * onp.random.normal(size=Y[500:550:2, :].shape)
    return T, Y_obs, freqs, onp, phase, plt, tec

if __name__ == '__main__':
    debug_param_search()