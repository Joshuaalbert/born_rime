from jax import numpy as jnp, numpy as np
from jax import vmap
from jax.scipy.linalg import solve_triangular
from jax.scipy.signal import _convolve_nd

###
# for soft_pmap
from typing import Callable, Optional
from warnings import warn
from jax import linear_util as lu
from jax.api_util import (wraps, flatten_fun, flatten_axes)
from jax.tree_util import (tree_flatten, tree_unflatten)
from jax.interpreters import pxla
from jax.api import (AxisName, _check_callable, _TempAxisName, _mapped_axis_size, _check_args, _reshape_split,
                     _reshape_merge, pmap)




def windowed_mean(a, w, mode='reflect'):
    dims = len(a.shape)
    a = a
    kernel = jnp.reshape(jnp.ones(w)/w, [w]+[1]*(dims-1))
    _w1 = (w-1)//2
    _w2 = _w1 if (w%2 == 1) else _w1 + 1
    pad_width = [(_w1, _w2)] + [(0,0)]*(dims-1)
    a = jnp.pad(a, pad_width=pad_width, mode=mode)
    return _convolve_nd(a,kernel, mode='valid', precision=None)


batched_diag = vmap(jnp.diag, 0, 0)

batched_multi_dot = vmap(jnp.linalg.multi_dot, 0, 0)


def constrain(v, a, b):
    return a + (jnp.tanh(v) + 1) * (b - a) / 2.

def constrain_std(v, vmin = 1e-3):
    return jnp.abs(v) + vmin

def deconstrain_std(v, vmin = 1e-3):
    return jnp.maximum(v - vmin, 0.)


def deconstrain(v, a, b):
    return jnp.arctanh(jnp.clip((v - a) * 2. / (b - a) - 1., -0.999, 0.999))

def constrain_tec(v, vmin = 0.5):
    return jnp.abs(v) + vmin

def deconstrain_tec(v, vmin = 0.5):
    return v - vmin

def constrain_omega(v, lower=0.5, scale=10.):
    return scale * jnp.log(jnp.exp(v) + 1.) + lower

def deconstrain_omega(v, lower=0.5, scale=10.):
    y = jnp.maximum(jnp.exp((v - lower) / scale) - 1., 0.)
    return jnp.maximum(-1e3, jnp.log(y))

def constrain_sigma(v, lower=0.01, scale=0.5):
    return scale * jnp.log(jnp.exp(v) + 1.) + lower

def deconstrain_sigma(v, lower=0.01, scale=0.5):
    y = jnp.maximum(jnp.exp((v - lower) / scale) - 1., 0.)
    return jnp.maximum(-1e3, jnp.log(y))



###
# Our own modification of soft_pmap
# TODO: remove after https://github.com/google/jax/issues/3400 is solved and pushed to pip

def soft_pmap(fun: Callable, axis_name: Optional[AxisName] = None, *,
              in_axes=0, backend: Optional[str] = None) -> Callable:
    warn("soft_pmap is an experimental feature and probably has bugs!")
    _check_callable(fun)
    axis_name = _TempAxisName(fun) if axis_name is None else axis_name

    @wraps(fun)
    def f_pmapped(*args, **kwargs):
        f = lu.wrap_init(fun)
        args_flat, in_tree = tree_flatten((args, kwargs))
        in_axes_flat = flatten_axes(in_tree, (in_axes, 0))
        assert all(axis in (0, None) for axis in in_axes_flat), \
            "soft_pmap currently only supports mapping over the leading axis"
        mapped_invars = tuple(axis is not None for axis in in_axes_flat)
        axis_size = _mapped_axis_size(in_tree, args_flat, in_axes_flat, "soft_pmap")
        _check_args(args_flat)
        flat_fun, out_tree = flatten_fun(f, in_tree)

        chunk_size, leftover = divmod(axis_size, pxla.unmapped_device_count(backend))
        if chunk_size == 0 and leftover:
            return pmap(fun, axis_name, in_axes=in_axes, backend=backend)(*args)  # can map directly onto hardware
        elif leftover:
            msg = ("soft_pmap mapped axis size must be divisible by the number of "
                   "XLA devices (or be less than or equal to that number), but got "
                   "an axis size of {} with {} devices.")
            raise ValueError(msg.format(axis_size, pxla.unmapped_device_count()))
        num_chunks = axis_size // chunk_size

        reshaped_args = [_reshape_split(num_chunks, x) if mapped else x for x, mapped in zip(args_flat, mapped_invars)]
        soft_mapped_fun = pxla.split_axis(flat_fun, axis_name, chunk_size)
        reshaped_outs = pxla.xla_pmap(soft_mapped_fun,
                                      *reshaped_args,
                                      backend=backend,
                                      axis_name=axis_name,
                                      axis_size=num_chunks,
                                      global_axis_size=None,
                                      devices=None,
                                      name=soft_mapped_fun.__name__,
                                      mapped_invars=mapped_invars)
        outs = [_reshape_merge(out) for out in reshaped_outs]
        return tree_unflatten(out_tree(), outs)

    namestr = "soft_pmap({}, axis_name={})".format
    f_pmapped.__name__ = namestr(f_pmapped.__name__, axis_name)
    return f_pmapped


def scalar_KL(mean, uncert, mean_prior, uncert_prior):
    """
    mean, uncert : [M]
    mean_prior,uncert_prior: [M]
    :return: scalar
    """
    # Get KL
    q_var = np.square(uncert)
    var_prior = np.square(uncert_prior)
    trace = q_var / var_prior
    mahalanobis = np.square(mean - mean_prior) / var_prior
    constant = -1.
    logdet_qcov = np.log(var_prior / q_var)
    twoKL = mahalanobis + constant + logdet_qcov + trace
    prior_KL = 0.5 * twoKL
    return np.sum(prior_KL)


def mvn_kl(mu_a, L_a, mu_b, L_b):
    def squared_frobenius_norm(x):
        return np.sum(np.square(x))

    b_inv_a = solve_triangular(L_b, L_a, lower=True)
    kl_div = (
            np.sum(np.log(np.diag(L_b))) - np.sum(np.log(np.diag(L_a))) +
            0.5 * (-L_a.shape[-1] +
                   squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                solve_triangular(L_b, mu_b[:, None] - mu_a[:, None], lower=True))))
    return kl_div


def fill_triangular(x, upper=False):
    m = x.shape[-1]
    if len(x.shape) != 1:
        raise ValueError("Only handles 1D to 2D transformation, because tril/u")
    m = np.int32(m)
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
        raise ValueError('Input right-most shape ({}) does not '
                         'correspond to a triangular matrix.'.format(m))
    n = np.int32(n)
    final_shape = list(x.shape[:-1]) + [n, n]
    if upper:
        x_list = [x, np.flip(x[..., n:], -1)]

    else:
        x_list = [x[..., n:], np.flip(x, -1)]
    x = np.reshape(np.concatenate(x_list, axis=-1), final_shape)
    if upper:
        x = np.triu(x)
    else:
        x = np.tril(x)
    return x


def fill_triangular_inverse(x, upper=False):
    n = x.shape[-1]
    n = np.int32(n)
    m = np.int32((n * (n + 1)) // 2)
    final_shape = list(x.shape[:-2]) + [m]
    if upper:
        initial_elements = x[..., 0, :]
        triangular_portion = x[..., 1:, :]
    else:
        initial_elements = np.flip(x[..., -1, :], axis=-2)
        triangular_portion = x[..., :-1, :]
    rotated_triangular_portion = np.flip(
        np.flip(triangular_portion, axis=-1), axis=-2)
    consolidated_matrix = triangular_portion + rotated_triangular_portion
    end_sequence = np.reshape(
        consolidated_matrix,
        list(x.shape[:-2]) + [n * (n - 1)])
    y = np.concatenate([initial_elements, end_sequence[..., :m - n]], axis=-1)
    return y