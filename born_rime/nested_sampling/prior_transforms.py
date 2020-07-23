from jax import numpy as jnp
from jax.scipy.special import ndtri


class PriorTransform(object):
    def __init__(self, mu, gamma):
        self.mu = mu
        self.gamma = gamma

    @property
    def ndims(self):
        mu, gamma = jnp.broadcast_arrays(self.mu, self.gamma)
        return mu.size

    def __call__(self, U):
        return ndtri(U) * self.gamma + self.mu