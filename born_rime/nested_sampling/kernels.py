from numpyro.infer.mcmc import MCMCKernel, HMCState, SAState, NUTS
from numpyro.infer.hmc_util import velocity_verlet, IntegratorState
from numpyro.infer.util import log_likelihood
from numpyro.util import copy_docs_from
from typing import NamedTuple, Union, Dict
import jax.numpy as jnp
from jax import random, grad, value_and_grad, tree_multimap
from jax.lax import cond


class UncalibratedState(NamedTuple):
    inner_state: Union[HMCState, SAState]#State from inner kernel
    extra: Dict#dict, can contain log_acceptance_correction for MH kernel


class ComposableKernel(MCMCKernel):
    """
    This is the base class for kernels that make use of an internal MCMCKernel.
    """

    def __init__(self, inner_kernel: MCMCKernel):
        self.inner_kernel = inner_kernel

    @copy_docs_from(MCMCKernel.postprocess_fn)
    def postprocess_fn(self, model_args, model_kwargs):
        return self.inner_kernel.postprocess_fn(model_args, model_kwargs)

    @copy_docs_from(MCMCKernel.init)
    def init(self, rng_key, num_warmup, init_params=None, model_args=None, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = dict()
        if model_args is None:
            model_args = tuple()
        self.inner_kernel.init(rng_key, num_warmup, init_params, model_args, model_kwargs)


class MetropolisHastings(ComposableKernel):
    def __init__(self, inner_kernel: MCMCKernel):
        super(MetropolisHastings, self).__init__(inner_kernel)

    @copy_docs_from(MCMCKernel.sample)
    def sample(self, state, model_args, model_kwargs):
        proposed_state = self.inner_kernel.sample(state, model_args, model_kwargs)

        if not isinstance(proposed_state, UncalibratedState):
            log_acceptance_correction = 0.
        else:
            proposed_state, extra = proposed_state
            log_acceptance_correction = extra.get('log_acceptance_correction', 0.)

        log_accept_ratio = proposed_state.potential_energy - state.potential_energy + log_acceptance_correction

        rng_key, rng_key_mh = random.split(proposed_state.rng_key, 2)
        log_uniform = jnp.log(random.uniform(rng_key_mh))
        is_accepted = log_uniform < log_accept_ratio
        new_state = cond(is_accepted,
                         proposed_state, lambda x: x,
                         state, lambda x: x)

        new_state = new_state._replace(i=state.i + 1, rng_key=rng_key)

        return new_state


class HardLikelihoodConstraint(ComposableKernel):
    """
    This augments a proposal state with log_acceptance_correction suitable for nested sampling exploration.
    """

    def __init__(self, inner_kernel: MCMCKernel, constraint, model):
        super(HardLikelihoodConstraint, self).__init__(inner_kernel)
        self.constraint = constraint
        self.model = model

    @copy_docs_from(MCMCKernel.sample)
    def sample(self, state, model_args, model_kwargs):
        proposed_state = self.inner_kernel.sample(state, model_args, model_kwargs)

        if isinstance(proposed_state, UncalibratedState):
            proposed_state, extra = proposed_state

        log_likelihood_proposed = log_likelihood(self.model, proposed_state.z, *model_args, **model_kwargs)
        log_acceptance_correction = jnp.where(log_likelihood_proposed > self.constraint,
                                              state.potential_energy - proposed_state.potential_energy,
                                              -jnp.inf)
        new_state = UncalibratedState(proposed_state, dict(log_acceptance_correction=log_acceptance_correction))
        return new_state


class MCMCKernelState(NamedTuple):
    i: int  # sample number, reset after warmup
    z: Dict[jnp.ndarray]  # unconstrained state
    z_grad: Dict[jnp.ndarray]  # gradient of targent log_prob at z
    potential_energy: float  # target log_prob
    energy: float  # canonical distribution log_prob
    num_steps: int  # number of intgration steps taken
    accept_prob: float  # Acceptance probability of the proposal.
    # Note that ``z`` does not correspond to the proposal if it is rejected.
    mean_accept_prob: float  # Mean acceptance probability until current iteration
    # during warmup adaptation or sampling (for diagnostics).
    diverging: bool  # A boolean value to indicate whether the current trajectory is diverging.
    adapt_state: Dict[jnp.ndarray]  # A ``HMCAdaptState`` namedtuple which contains adaptation information during warmup
    # + **step_size** - Step size to be used by the integrator in the next iteration.
    # + **inverse_mass_matrix** - The inverse mass matrix to be used for the next iteration.
    # + **mass_matrix_sqrt** - The square root of mass matrix to be used for the next iteration. In case of dense mass,
    # this is the Cholesky factorization of the mass matrix.
    rng_key: jnp.ndarray  # random number generator seed used for the iteration.


def constrained_velocity_verlet(potential_fn, kinetic_fn, constraint_fn):
    r"""
    Second order symplectic integrator that uses the velocity verlet algorithm
    for position `z` and momentum `r`, and additionally performs bounces from infitie potential walls.

    :param potential_fn: Python callable that computes the potential energy
        given input parameters. The input parameters to `potential_fn` can be
        any python collection type.
    :param kinetic_fn: Python callable that returns the kinetic energy given
        inverse mass matrix and momentum.
    :param constraint_fn: Python callable that returns the constraint C(z). A particle bounces when C(z) <= 0.
    :return: a pair of (`init_fn`, `update_fn`).

    [1] Nested Sampling with Constrained Hamiltonian Monte Carlo, Michael Betancourt, 2010, page 13
        (https://arxiv.org/pdf/1005.0157.pdf)
    """

    init_fn, no_bounce_update_fn = velocity_verlet(potential_fn, kinetic_fn)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        :param float step_size: Size of a single step.
        :param inverse_mass_matrix: Inverse of mass matrix, which is used to
            calculate kinetic energy.
        :param state: Current state of the integrator.
        :return: new state for the integrator.
        """
        z, r, _, z_grad = state
        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1/2)
        r_grad = grad(kinetic_fn, argnums=1)(inverse_mass_matrix, r)
        z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)
        potential_energy, z_grad = value_and_grad(potential_fn)(z)

        def do_bounce(r, z):
            grad_C = grad(constraint_fn)(z)
            n = grad_C / jnp.linalg.norm(grad_C)

            def _apply(r, grad_C, z_grad):
                r = r + 0.5 * step_size * z_grad  # undo last update
                return r - 2.0 * (r @ grad_C) * grad_C / jnp.sum(jnp.square(grad_C))

            r = tree_multimap(_apply, r, grad_C, z_grad)

        r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1)
        return IntegratorState(z, r, potential_energy, z_grad)

    def update_fn(step_size, inverse_mass_matrix, state):
        """
        :param float step_size: Size of a single step.
        :param inverse_mass_matrix: Inverse of mass matrix, which is used to
            calculate kinetic energy.
        :param state: Current state of the integrator.
        :return: new state for the integrator.
        """
        no_bounce_state = no_bounce_update_fn(step_size, inverse_mass_matrix, state)
        z, r, potential_energy, z_grad = no_bounce_state

        # r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1/2)
        # r_grad = grad(kinetic_fn, argnums=1)(inverse_mass_matrix, r)
        #
        #
        #
        # z = tree_multimap(lambda z, r_grad: z + step_size * r_grad, z, r_grad)  # z(n+1)


        C = constraint_fn(z)
        is_bounce = C <= 0.

        def do_bounce(r, z):
            grad_C = grad(constraint_fn)(z)
            n = grad_C/jnp.linalg.norm(grad_C)
            def _apply(r, grad_C, z_grad):
                r = r + 0.5 * step_size * z_grad #undo last update
                return r - 2.0 * (r @ grad_C) * grad_C / jnp.sum(jnp.square(grad_C))

            r = tree_multimap(_apply, r, grad_C, z_grad)

        r = cond(is_bounce, (r,z), do_bounce, (r,), lambda x: x)

        bounce_state = no_bounce_state._replace(r=r, potential_energy=-jnp.inf)

        # potential_energy, z_grad = value_and_grad(potential_fn)(z)
        # r = tree_multimap(lambda r, z_grad: r - 0.5 * step_size * z_grad, r, z_grad)  # r(n+1)

        return bounce_state

    return init_fn, update_fn


class ConstrainedNUTS(NUTS):

    @copy_docs_from(MCMCKernel.sample)
    def sample(self, state, model_args, model_kwargs):
        """
        Given the current `state`, return the next `state` using the given
        transition kernel.

        :param state: Arbitrary data structure representing the state for the
            kernel. For HMC, this is given by :data:`~numpyro.infer.mcmc.HMCState`.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state`.
        """
        raise NotImplementedError