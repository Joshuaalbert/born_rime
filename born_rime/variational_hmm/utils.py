from jax import numpy as jnp
from jax import vmap
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
