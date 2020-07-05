import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as onp
from scipy.stats import gaussian_kde

import numpyro
numpyro.set_host_device_count(6)

from jax import lax, random
import jax.numpy as np


import numpyro.distributions as dist
from numpyro.distributions.util import logsumexp
from numpyro.infer import MCMC, NUTS

freqs = np.linspace(121e6, 168e6,24)
tec_conv = -8.4479745e6 / freqs



def simulate_data(rng_key, seq_length):
    rng_dW, rng_obs_real, rng_obs_imag = random.split(rng_key, 3)

    omega_prior = 10.*np.ones(seq_length)
    sigma_prior = 0.2*np.ones([seq_length, freqs.size])
    dW = dist.Normal().sample(key=rng_dW, sample_shape=(seq_length,))
    tec = np.cumsum(omega_prior * dW)
    phases = tec[:,None] * tec_conv
    Y = np.exp(1j*phases) + sigma_prior * dist.Normal().sample(key=rng_obs_real, sample_shape=phases.shape) + 1j* sigma_prior * dist.Normal().sample(key=rng_obs_real, sample_shape=phases.shape)

    return (tec, Y)

def supervised_hmm(seq_length, Y_obs):
    sample_shape = (seq_length, freqs.size)
    X0 = numpyro.sample('X0', dist.Normal(0., 100.), sample_shape=(1,))
    omega_prior = numpyro.sample('omega_prior', dist.Uniform(1.,20.), sample_shape=(1,))
    dW = numpyro.sample('dW', dist.Normal(),sample_shape=(seq_length,))
    tec = X0 + omega_prior * np.cumsum(dW)
    phase = tec[:,None] * tec_conv

    sigma_real = numpyro.sample('sigma_real', dist.Uniform(0.01,1.))
    sigma_imag = numpyro.sample('sigma_imag', dist.Uniform(0.01,1.))
    Yreal = numpyro.sample('Yreal', dist.Normal(np.cos(phase), sigma_real), obs=Y_obs.real, sample_shape=sample_shape)
    Yimag = numpyro.sample('Yimag', dist.Normal(np.sin(phase), sigma_imag), obs=Y_obs.imag, sample_shape=sample_shape)

def main(args):
    print('Simulating data...')
    tec, Y_obs = simulate_data(
        random.PRNGKey(1),
        seq_length=100
    )
    print('Starting inference...')
    rng_key = random.PRNGKey(2)
    start = time.time()
    kernel = NUTS(supervised_hmm)
    mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=6,
                progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True)
    mcmc.run(rng_key, 100, Y_obs)
    samples = mcmc.get_samples()
    tec_rec = samples['X0'] + samples['omega_prior']*np.cumsum(samples['dW'], axis=-1)
    plt.plot(np.mean(tec_rec, 0))
    plt.show()
    plt.plot(tec)
    plt.show()
    plt.plot(tec - np.mean(tec_rec, 0))
    plt.show()
    print('\nMCMC elapsed time:', time.time() - start)


if __name__ == '__main__':
    assert numpyro.__version__.startswith('0.2.4')
    parser = argparse.ArgumentParser(description='Semi-supervised Hidden Markov Model')
    parser.add_argument('--num-categories', default=10, type=int)
    parser.add_argument('--num-words', default=100, type=int)
    parser.add_argument('--num-supervised', default=100, type=int)
    parser.add_argument('--num-unsupervised', default=500, type=int)
    parser.add_argument('-n', '--num-samples', nargs='?', default=1000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=500, type=int)
    parser.add_argument("--num-chains", nargs='?', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)