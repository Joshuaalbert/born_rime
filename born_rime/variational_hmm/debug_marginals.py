import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from jax import random, jit
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

from jax import lax
from numpyro.infer import ELBO, MCMC, NUTS, SVI, RenyiELBO
from numpyro.infer.autoguide import AutoBNAFNormal
from numpyro.infer.reparam import NeuTraReparam
import numpyro.optim as optim

def simulate_data(rng_key):

    freqs = jnp.linspace(121e6,168e6,24)
    tec = -200.#random.uniform(rng_key, minval= -100.,maxval= 100.)
    const = jnp.pi#random.uniform(rng_key, minval=-jnp.pi, maxval=jnp.pi)
    print("Ground truth tec {} const {}".format(tec, const))
    tec_conv = -8.4479745e6 / freqs
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + 0.75*random.normal(rng_key, shape=Y.shape)
    Sigma = 0.75**2*jnp.eye(48)
    return Y_obs, Sigma, tec_conv

def mvn_tec_const(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))

    std_const = numpyro.sample('std_const', dist.Uniform(0.01, 1.))
    std_tec = numpyro.sample('std_tec', dist.Uniform(0.5, 100.))

    L_Gamma = numpyro.sample('L_Gamma', dist.LKJCholesky(2))

    tec_const = numpyro.sample('tec_const', dist.MultivariateNormal(jnp.zeros(2), scale_tril=L_Gamma))
    phase = std_tec*tec_const[0] * tec_conv + std_const*tec_const[1]
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def uniform_const_uniform_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const = numpyro.sample('const', dist.Uniform(-jnp.pi, jnp.pi))
    tec = numpyro.sample('tec', dist.Uniform(-300., 300.))
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def uniform_const_componens_uniform_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const_real = numpyro.sample('const_real', dist.Uniform(-jnp.pi, jnp.pi))
    const_imag = numpyro.sample('const_imag', dist.Uniform(-jnp.pi, jnp.pi))
    const = jnp.arctan2(const_imag,const_real)
    numpyro.sample('const', dist.Normal(const, 0.01))
    tec = numpyro.sample('tec', dist.Uniform(-300., 300.))
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def normal_const_uniform_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const = numpyro.sample('const', dist.Normal(0., jnp.pi))
    tec = numpyro.sample('tec', dist.Uniform(-300., 300.))
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def uniform_const_normal_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const = numpyro.sample('const', dist.Uniform(-jnp.pi, jnp.pi))
    tec = numpyro.sample('tec', dist.Normal(0., 100.))
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def normal_const_normal_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    const = numpyro.sample('const', dist.Normal(0., jnp.pi))
    tec = numpyro.sample('tec', dist.Normal(0., 300.))
    phase = tec * tec_conv + const
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)

def normal_tec(Y_obs, Sigma, tec_conv):
    sigma = jnp.sqrt(jnp.diag(Sigma))
    tec = numpyro.sample('tec', dist.Normal(0., 300.))
    phase = tec * tec_conv
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    numpyro.sample('Y', dist.Normal(Y, sigma), obs=Y_obs)


def print_results(posterior, model):
    print("Results for {}".format(model.__name__))
    print("Const : %iles {}".format(np.quantile(posterior['const'], [0.15, 0.5, 0.85], axis=0)))
    print("Const : mean {}".format(np.mean(posterior['const'])))
    print("TEC : {}".format(np.quantile(posterior['tec'], [0.15, 0.5, 0.85], axis=0)))
    print("TEC : mean {}".format(np.mean(posterior['tec'])))

    plt.hist(posterior['tec'], bins=100)
    plt.title('tec')
    plt.show()
    plt.hist(posterior['const'], bins=100)
    plt.title('const')
    plt.show()

    plt.hist2d(posterior['const'], posterior['tec'],bins=100)
    plt.xlabel('const')
    plt.ylabel('tec')
    plt.title(model.__name__)
    plt.show()

def wrap(phi):
    return jnp.arctan2(jnp.sin(phi), jnp.cos(phi))

def main(args):
    print('Simulating data...')
    (Y_obs, Sigma, tec_conv) = simulate_data(random.PRNGKey(1))
    print('Starting inference...')
    # pymc3_solution(Y_obs, Sigma, tec_conv)
    rng_key = random.PRNGKey(4)
    for model in [normal_const_normal_tec]:
        start = time.time()
        guide = AutoBNAFNormal(model, num_flows=2, hidden_factors=[10, 10])
        svi = SVI(model, guide, optim.Adam(0.002), ELBO(num_particles=4), Y_obs=Y_obs, Sigma=Sigma, tec_conv=tec_conv)
        svi_state = svi.init(random.PRNGKey(2))
        last_state, losses = lax.scan(lambda state, i: svi.update(state), svi_state, jnp.zeros(20000))
        plt.plot(losses)
        plt.title("SVI loss")
        plt.show()

        guide_samples = guide.sample_posterior(random.PRNGKey(2), svi.get_params(last_state), sample_shape=(10000,))
        plt.hist(guide_samples['const'],bins=100)
        plt.title("Variational posterior: const")
        plt.show()

        plt.hist(guide_samples['tec'], bins=100)
        plt.title("Variational posterior: tec")
        plt.show()






        neutra = NeuTraReparam(guide, svi.get_params(last_state))
        #
        # guide_base_samples = dist.Normal(jnp.zeros(2), 1.).sample(random.PRNGKey(4), (1000,))
        # guide_trans_samples = neutra.transform_sample(guide_base_samples)
        #
        # plt.hist(guide_trans_samples['const'], bins=100)
        # plt.title("Guide samples: const")
        # plt.show()
        #
        # plt.hist(guide_trans_samples['tec'], bins=100)
        # plt.title("Guide samples: tec")
        # plt.show()


        kernel = NUTS(neutra.reparam(model), target_accept_prob=args.target_prob, init_strategy=numpyro.infer.init_to_prior())
        mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains, progress_bar=False)
        mcmc.run(rng_key, Y_obs, Sigma, tec_conv)
        zs = mcmc.get_samples(group_by_chain=True)["auto_shared_latent"]
        samples = neutra.transform_sample(zs)

        print("Results with NeuTra")
        numpyro.diagnostics.print_summary(samples)
        samples = {k:v.reshape((v.shape[0]*v.shape[1],)+v.shape[2:]) for k,v in samples.items()}
        if model is mvn_tec_const:
            samples['tec'] = samples['tec_const'][:,0]
            samples['const'] = samples['tec_const'][:,1]
        samples['const'] = wrap(samples['const'])
        print_results(samples, model)
        print('\nMCMC elapsed time:', time.time() - start)

        # start = time.time()
        # kernel = NUTS(model, target_accept_prob=args.target_prob,
        #               init_strategy=numpyro.infer.init_to_prior())
        # mcmc = MCMC(kernel, args.num_warmup, args.num_samples, num_chains=args.num_chains, progress_bar=False)
        # mcmc.run(rng_key, Y_obs, Sigma, tec_conv)
        # samples = mcmc.get_samples(group_by_chain=True)
        # print("Results without NeuTra")
        # numpyro.diagnostics.print_summary(samples)
        # samples = {k: v.reshape((v.shape[0] * v.shape[1],) + v.shape[2:]) for k, v in samples.items()}
        # if model is mvn_tec_const:
        #     samples['tec'] = samples['tec_const'][:, 0]
        #     samples['const'] = samples['tec_const'][:, 1]
        # print_results(samples, model)
        # print('\nMCMC elapsed time:', time.time() - start)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get copula of const.')
    parser.add_argument('-n', '--num-samples', nargs='?', default=10000, type=int)
    parser.add_argument('--num-warmup', nargs='?', default=10000, type=int)
    parser.add_argument("--num-chains", nargs='?', default=24, type=int)
    parser.add_argument("--target-prob", nargs='?', default=0.8, type=float)
    parser.add_argument('--device', default='cpu', type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)