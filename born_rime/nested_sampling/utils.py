import jax.numpy as jnp
from jax import random
from jax.random import split, gumbel, randint, permutation, uniform
import numpy as np


def random_ortho_matrix(key, n):
    """
    Samples a random orthonormal num_parent,num_parent matrix from Stiefels manifold.

    From https://stackoverflow.com/a/38430739

    """
    H = random.normal(key, shape=(n, n))
    Q, R = jnp.linalg.qr(H)
    Q = Q @ jnp.diag(jnp.sign(jnp.diag(R)))
    return Q

def test_random_ortho_normal_matrix():
    H = random_ortho_matrix(random.PRNGKey(0), 3)
    assert jnp.all(jnp.isclose(H @ H.conj().T, jnp.eye(3), atol=1e-7))

