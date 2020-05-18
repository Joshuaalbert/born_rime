import jax.numpy as jnp
import jax
from jax.lax import while_loop, cond
from typing import NamedTuple

LineSearchResults = NamedTuple('LineSearchResults',
                               [('failed', bool),  # Were both Wolfe criteria satisfied
                                ('nfev', int),  # Number of functions evaluations
                                ('ngev', int),  # Number of gradients evaluations
                                ('k', int),  # Number of iterations
                                ('a_k', float),  # Step size
                                ('f_k', jnp.ndarray),  # Final function value
                                ('g_k', jnp.ndarray)  # Final gradient value
                                ])


def line_search_backtracking(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4, c2=0.9):
    """
    Performs an inexact line-search. It is a modified backtracking line search. Instead of reducing step size by a
    number < 1 if Wolfe conditions are not met, we check the sign of  u = del_t restricted_func(t).
    If u > 0 then we do normal backtrack, otherwise we search forward. Normal backtracking can fail to satisfy strong
    Wolfe conditions. This extra step costs one extra gradient evaluation. For explaination see figures 3.1--3.4 in
    https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

    The original backtracking algorithim is p. 37.

    Args:
        value_and_gradient: function and gradient
        position: position to search from
        direction: descent direction to search along
        f_0: optionally give starting function value at position
        g_0: optionally give starting gradient at position
        max_iterations: maximum number of searches
        c1, c2: Wolfe criteria numbers from above reference

    Returns: LineSearchResults

    """

    def restricted_func(t):
        return value_and_gradient(position + t * direction)

    grad_restricted = jax.grad(lambda t: restricted_func(t)[0])

    state = LineSearchResults(failed=jnp.array(True), nfev=0, ngev=0, k=0, a_k=1., f_k=None, g_k=None)
    rho_neg = 0.8
    rho_pos = 1.2

    if f_0 is None or g_0 is None:
        f_0, g_0 = value_and_gradient(position)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
    state = state._replace(f_k=f_0, g_k=g_0)

    def body(state):
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
        # Wolfe 1 (3.6a)
        wolfe_1 = f_kp1 <= state.f_k + c1 * state.a_k * jnp.dot(state.g_k, direction)
        # Wolfe 2 (3.7b)
        wolfe_2 = jnp.abs(jnp.dot(g_kp1, direction)) <= c2 * jnp.abs(jnp.dot(state.g_k, direction))

        state = state._replace(failed=~(wolfe_1 & wolfe_2), k=state.k + 1)

        def backtrack(state):
            # TODO: it may make sense to only do this once on the first iteration.
            # Moreover, can this be taken out of cond?
            u = grad_restricted(state.a_k)
            state = state._replace(ngev=state.ngev + 1)
            # state = state._replace(a_k=cond(u > 0, None, lambda *x: state.a_k * rho_neg,
            # None, lambda *x: state.a_k * rho_pos))
            a_k = state.a_k * jnp.where(u > 0, rho_neg, rho_pos)
            state = state._replace(a_k=a_k)
            return state

        def finish(args):
            state, f_kp1, g_kp1 = args
            state = state._replace(f_k=f_kp1, g_k=g_kp1)
            return state

        state = cond(state.failed, state, backtrack, (state, f_kp1, g_kp1), finish)

        return state

    state = while_loop(lambda state: state.failed & (state.k < max_iterations),
                       body,
                       state
                       )

    def maybe_update(state):
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(f_k=f_kp1, g_k=g_kp1, nfev=state.nfev + 1, ngev=state.ngev + 1)
        return state

    state = cond(state.failed, state, maybe_update, state, lambda state: state)

    return state


def cubic_interpolation(a_1, phi_1, dphi_1, a_2, phi_2, dphi_2):
    """
    Computes the cubic interpolation minimiser based on two previous evaluations of the
    restricted function. It assumes that the minimum of the interpolation is in (a_1,a_2).

    Args:
        a_1: a previous step
        phi_1: phi(a_1)
        dphi_1: phi'(a_1)
        a_2: a previous step not equal to a_1
        phi_2: phi(a_2)
        dphi_2: phi'(a_2)

    Returns: a step minimising the cubic interpolant.

    """
    # return (a_1 + a_2)/2.
    d1 = dphi_1 + dphi_2 - 3. * (phi_1 - phi_2) / (a_1 - a_2)
    d2 = jnp.sign(a_2 - a_1) * jnp.sqrt(d1 ** 2 - dphi_1 * dphi_2)
    a_3 = a_1 - (a_2 - a_1) * (dphi_2 + d2 - d1) / (dphi_2 - dphi_1 + 2. * d2)
    return a_3

def _cubicmin(a,fa,fpa,b,fb,c,fc):
    C = fpa
    db = b-a
    dc = c-a
    denom = (db * dc) ** 2 * (db - dc)
    d1 = jnp.array([[dc**2, -db**2],
                   [-dc**3, db ** 3]])
    A, B = jnp.dot(d1, jnp.array([fb - fa - C * db, fc - fa - C * dc])) / denom

    radical = B * B - 3. * A * C
    xmin = a + (-B + jnp.sqrt(radical)) / (3. * A)

    return xmin

def _quadmin(a, fa, fpa, b, fb):
    D = fa
    C = fpa
    db = b - a
    B = (fb - D - C * db) / (db ** 2)
    xmin = a - C / (2. * B)
    return xmin


def _binary_replace(replace_bit, original_dict, new_dict, keys):
    """
    Similar to np.where, but fewer ops.
    """
    not_replace_bit = ~replace_bit
    out = dict()
    for key in keys:
        out[key] = not_replace_bit * original_dict[key] + replace_bit * new_dict[key]
    return out


def _zoom(restricted_func_and_grad, wolfe_one, wolfe_two, a_lo, phi_lo, dphi_lo, a_hi, phi_hi, dphi_hi, g_0, pass_through):
    """
    Algorithm 3.6 from
    https://pages.mtu.edu/~struther/Courses/OLD/Sp2013/5630/Jorge_Nocedal_Numerical_optimization_267490.pdf

    Args:
        restricted_func_and_grad:
        wolfe_one:
        wolfe_two:
        a_lo:
        phi_lo:
        dphi_lo:
        a_hi:
        phi_hi:
        dphi_hi:
        g_0:
        pass_through:

    Returns:

    """
    ZoomState = NamedTuple('ZoomState',
                           [('done', bool),
                            ('j', int),
                            ('a_lo', float),
                            ('phi_lo', float),
                            ('dphi_lo', float),
                            ('a_hi', float),
                            ('phi_hi', float),
                            ('dphi_hi', float),
                            ('a_rec', float),
                            ('phi_rec', float),
                            ('a_star', float),
                            ('phi_star', float),
                            ('dphi_star', float),
                            ('g_star', float),
                            ('nfev', int),
                            ('ngev', int)])

    state = ZoomState(done=False,
                      j=0,
                      a_lo=a_lo,
                      phi_lo=phi_lo,
                      dphi_lo=dphi_lo,
                      a_hi=a_hi,
                      phi_hi=phi_hi,
                      dphi_hi=dphi_hi,
                      a_rec = (a_lo+a_hi)/2.,
                      phi_rec = (phi_lo + phi_hi)/2.,
                      a_star=1.,
                      phi_star=phi_lo,
                      dphi_star=dphi_lo,
                      g_star=g_0,
                      nfev=0,
                      ngev=0
                      )

    delta1 = 0.2
    delta2 = 0.1
    def body(state):
        """
        Body of zoom algorithm. We use boolean arithmatic to avoid using jax.cond so that it works on GPU/TPU.
        """
        a = jnp.minimum(state.a_hi, state.a_lo)
        b = jnp.maximum(state.a_hi, state.a_lo)
        dalpha = (b - a)
        cchk = delta1 * dalpha
        qchk = delta2 * dalpha

        state = state._replace(done=state.done | (dalpha == 0.))

        a_j_cubic = _cubicmin(state.a_lo, state.phi_lo, state.dphi_lo, state.a_hi, state.phi_hi, state.a_rec, state.phi_rec)
        use_cubic = (state.j > 0) & (a_j_cubic > a + cchk) & (a_j_cubic < b - cchk)
        a_j_quad = _quadmin(state.a_lo, state.phi_lo, state.dphi_lo, state.a_hi, state.phi_hi)
        use_quad = (~use_cubic) &  (a_j_quad > a + qchk) & (a_j_quad < b - qchk)
        a_j_bisection = (state.a_lo + state.a_hi)/2.
        use_bisection = (~use_cubic) & (~use_quad)

        a_j = jnp.where(use_cubic, a_j_cubic, state.a_rec)
        a_j = jnp.where(use_quad, a_j_quad, a_j)
        a_j = jnp.where(use_bisection, a_j_bisection, a_j)


        # a_j = cubic_interpolation(state.a_lo, state.phi_lo, state.dphi_lo,
        #                           state.a_hi, state.phi_hi, state.dphi_hi)


        phi_j, dphi_j, g_j = restricted_func_and_grad(a_j)
        state = state._replace(nfev=state.nfev + 1,
                               ngev=state.ngev + 1)

        hi_to_j = wolfe_one(a_j, phi_j) | (phi_j >= state.phi_lo)
        star_to_j = wolfe_two(dphi_j) & (~hi_to_j)
        hi_to_lo = (dphi_j * (state.a_hi - state.a_lo) >= 0.) & (~hi_to_j) & (~star_to_j)
        lo_to_j = (~hi_to_j) & (~star_to_j)

        state = state._replace(**_binary_replace(hi_to_j,
                                                 state._asdict(),
                                                 dict(a_hi=a_j,
                                                      phi_hi=phi_j,
                                                      dphi_hi=dphi_j,
                                                      a_rec=state.a_hi,
                                                      phi_rec=state.phi_hi),
                                                 ['a_hi', 'phi_hi', 'dphi_hi','a_rec','phi_rec']))

        # for termination
        state = state._replace(done=star_to_j | state.done, **_binary_replace(star_to_j,
                                                                 state._asdict(),
                                                                 dict(a_star=a_j,
                                                                      phi_star=phi_j,
                                                                      dphi_star=dphi_j,
                                                                      g_star=g_j),
                                                                 ['a_star', 'phi_star', 'dphi_star','g_star']))

        state = state._replace(**_binary_replace(hi_to_lo,
                                                 state._asdict(),
                                                 dict(a_hi=a_lo,
                                                      phi_hi=phi_lo,
                                                      dphi_hi=dphi_lo,
                                                      a_rec=state.a_hi,
                                                      phi_rec=state.phi_hi),
                                                 ['a_hi', 'phi_hi', 'dphi_hi','a_rec', 'phi_rec']))

        state = state._replace(**_binary_replace(lo_to_j,
                                                 state._asdict(),
                                                 dict(a_lo=a_j,
                                                      phi_lo=phi_j,
                                                      dphi_lo=dphi_j,
                                                      a_rec=state.a_lo,
                                                      phi_rec=state.phi_lo),
                                                 ['a_lo', 'phi_lo', 'dphi_lo','a_rec','phi_rec']))

        state = state._replace(j=state.j + 1)
        return state

    state = while_loop(lambda state: (~state.done) & (~pass_through),
                       body,
                       state)

    # while (~state.done) & (~pass_through):
    #     state = body(state)

    return state


def line_search(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4,
                         c2=0.9):
    """
    Inexact line search that satisfies strong Wolfe conditions.
    Algorithm 3.5 from Wright and Nocedal, 'Numerical Optimization', 1999, pg. 59-60

    Notes:
        We utilise boolean arithmetic to avoid jax.cond calls which don't work on accelerators.
    Args:
        value_and_gradient:
        position:
        direction:
        f_0:
        g_0:
        max_iterations:
        c1:
        c2:

    Returns:

    """

    def restricted_func_and_grad(t):
        phi, g = value_and_gradient(position + t * direction)
        dphi = jnp.dot(g, direction)
        return phi, dphi, g

    LineSearchState = NamedTuple('LineSearchState',
                                 [('done', bool),
                                  ('failed', bool),
                                  ('i', int),
                                  ('a_i1', float),
                                  ('phi_i1', float),
                                  ('dphi_i1', float),
                                  ('nfev', int),
                                  ('ngev', int),
                                  ('a_star', float),
                                  ('phi_star', float),
                                  ('dphi_star', float),
                                  ('g_star', jnp.ndarray)])
    if f_0 is None or g_0 is None:
        phi_0, dphi_0, g_0 = restricted_func_and_grad(0.)
    else:
        phi_0 = f_0
        dphi_0 = jnp.dot(g_0, direction)

    def wolfe_one(a_i, phi_i):
        #actually negation of W1
        return phi_i > phi_0 + c1 * a_i * dphi_0

    def wolfe_two(dphi_i):
        return jnp.abs(dphi_i) <= -c2 * dphi_0

    state = LineSearchState(done=False,
                            failed=False,
                            i=1,
                            a_i1=0.,
                            phi_i1=phi_0,
                            dphi_i1=dphi_0,
                            nfev=1 if (f_0 is None or g_0 is None) else 0,
                            ngev=1 if (f_0 is None or g_0 is None) else 0,
                            a_star=0.,
                            phi_star=phi_0,
                            dphi_star=dphi_0,
                            g_star = g_0)


    def body(state):
        #no amax, we just double as in scipy.
        #unlike original algorithm we do our next choice at the start of this loop
        a_i = jnp.where(state.i == 1, 1., state.a_i1 * 2.)
        #if a_i <= 0 then something went wrong.
        state = state._replace(failed=a_i<=0.)




        phi_i, dphi_i, g_i = restricted_func_and_grad(a_i)
        state = state._replace(nfev=state.nfev + 1,
                               ngev=state.ngev + 1)

        #TODO: Check. In Scipy it says: ((phi_a1 >= phi_a0) and (i > 1))
        # but in [...] it says: (phi(a[i]) >= phi(a[i-1]) and (i > 1))
        # it appears okay though.
        star_to_zoom1 = wolfe_one(a_i, phi_i) | ((phi_i >= state.phi_i1) & (state.i > 1))
        star_to_i = wolfe_two(dphi_i) & (~star_to_zoom1)
        star_to_zoom2 = (dphi_i >= 0.) & (~star_to_zoom1) & (~star_to_i)

        zoom1 = _zoom(restricted_func_and_grad,
                      wolfe_one,
                      wolfe_two,
                      state.a_i1,
                      state.phi_i1,
                      state.dphi_i1,
                      a_i,
                      phi_i,
                      dphi_i,
                      g_0,
                      ~star_to_zoom1)

        state = state._replace(nfev=state.nfev + zoom1.nfev,
                               ngev=state.ngev + zoom1.ngev)

        zoom2 = _zoom(restricted_func_and_grad,
                      wolfe_one,
                      wolfe_two,
                      a_i,
                      phi_i,
                      dphi_i,
                      state.a_i1,
                      state.phi_i1,
                      state.dphi_i1,
                      g_0,
                      ~star_to_zoom2)

        state = state._replace(nfev=state.nfev + zoom2.nfev,
                               ngev=state.ngev + zoom2.ngev)

        state = state._replace(done=star_to_zoom1 | state.done, **_binary_replace(star_to_zoom1,
                                                                     state._asdict(),
                                                                     zoom1._asdict(),
                                                                     ['a_star', 'phi_star', 'dphi_star','g_star']))

        state = state._replace(done=star_to_i| state.done, **_binary_replace(star_to_i,
                                                                     state._asdict(),
                                                                     dict(a_star=a_i,
                                                                          phi_star=phi_i,
                                                                          dphi_star=dphi_i,
                                                                          g_star=g_i),
                                                                     ['a_star', 'phi_star', 'dphi_star','g_star']))

        state = state._replace(done=star_to_zoom2| state.done, **_binary_replace(star_to_zoom2,
                                                                     state._asdict(),
                                                                     zoom2._asdict(),
                                                                     ['a_star', 'phi_star', 'dphi_star','g_star']))

        state = state._replace(i=state.i + 1, a_i1=a_i, phi_i1=phi_i, dphi_i1=dphi_i)
        return state

    state = while_loop(lambda state: (~state.done) & (state.i <= max_iterations),
                       body,
                       state)

    # while (~state.done) & (state.i <= max_iterations):
    #     state = body(state)

    results = LineSearchResults(failed=~state.done,
                                nfev=state.nfev,
                                ngev=state.ngev,
                                k=state.i,
                                a_k=state.a_star,
                                f_k=state.phi_star,
                                g_k=state.g_star)
    return results

