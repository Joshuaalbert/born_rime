import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from numpyro import handlers
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist

from jax import lax
from numpyro.infer import ELBO, MCMC, NUTS, SVI, RenyiELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam
import numpyro.optim as optim


def simulate_data(rng_key):
    freqs = jnp.linspace(121e6, 168e6, 24)
    tec = -200.  # random.uniform(rng_key, minval= -100.,maxval= 100.)
    const = 0.  # jnp.pi#random.uniform(rng_key, minval=-jnp.pi, maxval=jnp.pi)
    print("Ground truth tec {} const {}".format(tec, const))
    tec_conv = -8.4479745e6 / freqs
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.25 * random.normal(rng_key, shape=Y.shape)
    Sigma = 0.25 ** 2 * jnp.eye(48)
    return Y_obs, Sigma, tec_conv


def switched_normal_const_normal_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const = numpyro.sample('const', dist.Normal(0., jnp.pi))
    tec = numpyro.sample('tec', dist.Normal(0., 300.))
    systematic_exists = numpyro.sample('systematic_exists', dist.Uniform())
    phase = tec * tec_conv + (systematic_exists < 0.5) * const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def print_results(posterior, model):
    print("Results for {}".format(model.__name__))

    plt.hist(posterior['tec'], bins=100)
    plt.title('tec')
    plt.show()

    plt.hist(posterior['const'], bins=100)
    plt.title('const')
    plt.show()

    plt.hist(posterior['systematic_exists'], bins=100)
    plt.title('systematic_exists')
    plt.show()

    plt.hist2d(posterior['const'], posterior['tec'], bins=100)
    plt.xlabel('const')
    plt.ylabel('tec')
    plt.title(model.__name__)
    plt.show()


def wrap(phi):
    return jnp.arctan2(jnp.sin(phi), jnp.cos(phi))


def log_likelihood(rng_key, params, model, *args, **kwargs):
    model = handlers.substitute(handlers.seed(model, rng_key), params)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    print(model_trace)
    obs_node = model_trace['Y']
    return obs_node['fn'].log_prob(obs_node['value'])


def log_predictive_density(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return jnp.sum(logsumexp(log_lk_vals, 0) - np.log(n))

def log_Z(rng_key, params, model, *args, **kwargs):
    n = list(params.values())[0].shape[0]
    log_lk_fn = vmap(lambda rng_key, params: log_likelihood(rng_key, params, model, *args, **kwargs))
    log_lk_vals = log_lk_fn(random.split(rng_key, n), params)
    return np.sum(logsumexp(log_lk_vals, 0) - np.log(n))


def main(args):
    print('Simulating data...')
    (Y_obs, Sigma, tec_conv) = simulate_data(random.PRNGKey(1))
    print('Starting inference...')
    rng_key = random.PRNGKey(4)
    for model in [switched_normal_const_normal_tec]:
        start = time.time()

        if False:
            guide = AutoBNAFNormal(model, num_flows=2, hidden_factors=[10, 10])
            svi = SVI(model, guide, optim.Adam(0.001), ELBO(num_particles=4), Y_obs=Y_obs, Sigma=Sigma,
                      tec_conv=tec_conv)
            svi_state = svi.init(random.PRNGKey(2))
            last_state, losses = lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(20000))
            plt.plot(losses)
            plt.title("SVI loss")
            plt.show()

            guide_samples = guide.sample_posterior(random.PRNGKey(2), svi.get_params(last_state), sample_shape=(10000,))
            plt.hist(guide_samples['const'], bins=100)
            plt.title("Variational posterior: const")
            plt.show()

            plt.hist(guide_samples['tec'], bins=100)
            plt.title("Variational posterior: tec")
            plt.show()

            neutra = NeuTraReparam(guide, svi.get_params(last_state))

            kernel = NUTS(neutra.reparam(model), target_accept_prob=args.target_prob,
                          init_strategy=numpyro.infer.init_to_prior())
            mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains, progress_bar=False)
            mcmc.run(rng_key, Y_obs, Sigma, tec_conv)
            zs = mcmc.get_samples(group_by_chain=True)["auto_shared_latent"]
            samples = neutra.transform_sample(zs)
            print("Results without NeuTra")
        else:
            kernel = NUTS(model, target_accept_prob=args.target_prob, init_strategy=numpyro.infer.init_to_prior())
            mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains, progress_bar=False)
            mcmc.run(rng_key, Y_obs, Sigma, tec_conv)
            samples = mcmc.get_samples(group_by_chain=True)

            print("Results with NeuTra")



        samples['const'] = wrap(samples['const'])
        samples['systematic_exists'] = jnp.array(samples['systematic_exists'] < 0.5, jnp.float32)
        numpyro.diagnostics.print_summary(samples)
        samples = {k: v.reshape((v.shape[0] * v.shape[1],) + v.shape[2:]) for k, v in samples.items()}

        log_density = log_predictive_density(random.PRNGKey(5), samples, model, Y_obs=Y_obs, Sigma=Sigma,
                                             tec_conv=tec_conv)

        print("log predictive density:", log_density)

        print_results(samples, model)
        print('\nMCMC elapsed time:', time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get copula of const.')
    parser.add_argument('-n', '--num-samples', nargs='?', default=20000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=10000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=24, type=int)
    parser.add_argument("--target-prob", nargs='?', default=0.8, type=float)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
