import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from numpyro import handlers
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
from typing import NamedTuple, Dict
import numpyro.distributions as dist

from jax import lax
from numpyro.infer import ELBO, MCMC, NUTS, SVI, RenyiELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam
import numpyro.optim as optim
from numpyro.infer.util import log_likelihood
from .kernels import MetropolisHastings, HardLikelihoodConstraint


def _log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    obs_node = model_trace['Y']
    return obs_node['fn'].log_prob(obs_node['value'])


def sample_prior(rng_key, model, sample_shape, *args, **kwargs):
    model = handlers.seed(model, rng_key)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    site_samples = {}
    for k, v in model_trace.items():
        if v['is_observed']:
            continue
        site_samples[k] = v['fn'].sample(sample_shape=sample_shape)
    return site_samples


def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - np.log(n))


class NestState(NamedTuple):
    active: Dict[jnp.ndarray]  # active set of points
    rng_key: jnp.ndarray  # PRNG key
    log_X_i1: float
    log_Z: float
    log_Lstar: float


class NestedSampling(object):
    def __init__(self, model, num_active, target_prob=0.8, num_chains=1, num_warmup=0, num_samples=1):
        self.model = model
        self.num_active = num_active
        self.target_prob = target_prob
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.num_samples = num_samples

    def sample(self, rng_key, *args, **kwargs):
        _batched_log_lk_vals = vmap(
            lambda rng_key, params: log_likelihood(rng_key, active, self.model, *args, **kwargs))
        active = sample_prior(rng_key, self.model, (self.num_active,), *args, **kwargs)
        state = NestState(active=active, rng_key=rng_key, log_X_i1=0., log_Z=0.)

        def body(state: NestState):
            log_lk_vals = _batched_log_lk_vals(random.split(rng_key, self.num_active), state.active)
            worst = jnp.argmin(log_lk_vals)
            best = jnp.argmax(log_lk_vals)
            log_Lstar = log_lk_vals[worst]
            keys = random.split(rng_key, 3)
            t_i = random.beta(keys[0], self.num_active, 1)
            log_X_i = state.log_X_i1 + jnp.log(t_i)
            log_w_i = jnp.log(jnp.exp(state.log_X_i1) - jnp.exp(log_X_i))
            # log_Z = jnp.exp(state.log_Z) + jnp.exp(log_Lstar + log_w_i)
            log_Z = jnp.logaddexp(state.log_Z, log_Lstar + log_w_i)
            # replace worst point with new sample
            kernel = NUTS(self.model, target_accept_prob=self.target_prob,
                          init_strategy=numpyro.infer.init_to_value({k: v[best] for k, v in active.items()}))
            mcmc = MCMC(kernel, self.num_warmup, self.num_samples, num_chains=1, progress_bar=False)
            mcmc.run(rng_key, *args, **kwargs)
            samples = mcmc.get_samples(group_by_chain=False)
            log_lk_vals = _batched_log_lk_vals(random.split(keys[1], self.num_samples), samples)


            state = state._replace(log_Z=log_Z, rng_key=keys[2], log_X_i1=log_X_i)
            return state
