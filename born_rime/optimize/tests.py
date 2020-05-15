import jax
from jax import numpy as np

from born_rime.optimize.line_search import line_search, LineSearchResults
from born_rime.optimize.bfgs_minimize import bfgs_minimize, BFGSResults


def test_line_search():
    def f(x):
        return np.sum(x) ** 3

    assert line_search(jax.value_and_grad(f), np.ones(2), np.array([0.5, -0.25])).failed

    def f(x):
        return np.sum(x) ** 3

    assert not line_search(jax.value_and_grad(f), np.ones(2), np.array([-0.5, -0.25])).failed

    v_g = jax.value_and_grad(f)
    @jax.jit
    def jit_line_search(position, direction):
        return line_search(v_g,position,direction )

    position, direction = np.ones(2), np.array([-0.5, -0.25])

    assert jit_line_search(position, direction).a_k == line_search_nojit(v_g, position, direction).a_k

    position, direction = np.ones(2), np.array([0.5, -0.25])

    assert np.isclose(jit_line_search(position, direction).a_k, line_search_nojit(v_g, position, direction).a_k)


def test_minimize():

    def rosenbrock(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)


    x0 = np.zeros(2)


    @jax.jit
    def min_op(x0):
        result = bfgs_minimize(rosenbrock, x0, options=dict(analytic_initial_hessian=True))
        return result

    jax_res1 = min_op(x0)

    print("Final with analytic initialisation:\n", jax_res1 )

    jax_res1_nojit = bfgs_minimize_nojit(rosenbrock, x0, options=dict(analytic_initial_hessian=True))

    assert np.all(jax_res1.x_k == jax_res1_nojit.x_k)

    @jax.jit
    def min_op(x0):
        result = bfgs_minimize(rosenbrock, x0, options=dict(analytic_initial_hessian=False))
        return result

    jax_res2 = min_op(x0)

    print("Final eye initialisation (like scipy):\n", jax_res2)

    from scipy.optimize import minimize as smin
    import numpy as onp

    def rosenbrock(x):
        return onp.sum(100. * onp.diff(x) ** 2 + (1. - x[:-1]) ** 2)

    scipy_res = smin(rosenbrock, x0, method='BFGS')
    print("Scipy:\n", scipy_res)

    assert np.all(np.isclose(scipy_res.x, jax_res1.x_k))
    assert np.all(np.isclose(scipy_res.x, jax_res2.x_k))


def line_search_nojit(value_and_gradient, position, direction, f_0=None, g_0=None, max_iterations=50, c1=1e-4, c2=0.9):
    def restricted_func(t):
        return value_and_gradient(position + t * direction)

    grad_restricted = jax.grad(lambda t: restricted_func(t)[0])

    state = LineSearchResults(failed=np.array(True), nfev=0, ngev=0, k=0, a_k=1., f_k=None, g_k=None)
    rho_neg = 0.8
    rho_pos = 1.2

    if f_0 is None or g_0 is None:
        f_0, g_0 = value_and_gradient(position)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
    state = state._replace(f_k=f_0, g_k=g_0)

    while state.failed and state.k < max_iterations:
        f_kp1, g_kp1 = restricted_func(state.a_k)
        # print(f_kp1, g_kp1)
        state = state._replace(nfev=state.nfev + 1, ngev=state.ngev + 1)
        # Wolfe 1 (3.6a)
        wolfe_1 = f_kp1 <= state.f_k + c1 * state.a_k * np.dot(state.g_k, direction)
        # Wolfe 2 (3.7b)
        wolfe_2 = np.abs(np.dot(g_kp1, direction)) <= c2 * np.abs(np.dot(state.g_k, direction))

        # print(state.a_k)
        # print('W1', f_kp1, state.f_k + c1 * state.a_k * np.dot(state.g_k, direction))
        # print('W2', np.abs(np.dot(g_kp1, direction)), c2 * np.abs(np.dot(state.g_k, direction)))

        state = state._replace(failed=~np.logical_and(wolfe_1, wolfe_2), k=state.k + 1)
        if not state.failed:
            # print('break')
            state = state._replace(f_k=f_kp1, g_k=g_kp1)
            break
        else:
            u = grad_restricted(state.a_k)
            state = state._replace(ngev=state.ngev + 1)
            if u > 0:
                state = state._replace(a_k=state.a_k * rho_neg)
            else:
                state = state._replace(a_k=state.a_k * rho_pos)

    if state.failed:
        f_kp1, g_kp1 = restricted_func(state.a_k)
        state = state._replace(f_k=f_kp1, g_k=g_kp1, nfev=state.nfev + 1, ngev=state.ngev + 1)
    return state


def bfgs_minimize_nojit(func, x0, options=None):
    """
        Minimisation with BFGS. Inexact line search is performed satisfying both Wolfe conditions
        (strong version of second one).
        Args:
            func: function to minimise, should take a single flat parameter, similar to numpy
            x0: initial guess of parameter
            options: A dictionary of solver options. All methods accept the following generic options:
                maxiter : int or None which means inf
                    Maximum number of iterations to perform. Depending on the
                    method each iteration may use several function evaluations.
                analytic_initial_hessian: bool,
                    whether to use JAX to get hessian for the initialisation, otherwise it's eye.
                g_tol: float
                    stopping criteria is |grad(func)|_2 \lt g_tol
                ls_maxiter: int or None which means inf
                    Maximum number of line search iterations to perform

        Returns: BFGSResults

    """
    if options is None:
        options = dict()
    maxiter: int = options.get('maxiter', None)
    analytic_initial_hessian: bool = options.get('analytic_initial_hessian', True)
    g_tol: float = options.get('g_tol', 1e-8)
    ls_maxiter: int = options.get('ls_maxiter', 50)

    state = BFGSResults(converged=False,
                        failed=False,
                        k=0,
                        nfev=0,
                        ngev=0,
                        nhev=0,
                        x_k=x0,
                        f_k=None,
                        g_k=None,
                        H_k=None)

    if maxiter is None:
        maxiter = np.inf

    D = x0.shape[0]

    if analytic_initial_hessian:
        hess = jax.hessian(func, argnums=0)
        initial_B = hess(x0)
        initial_H = np.linalg.pinv(initial_B)
        state = state._replace(nhev=state.nhev + 1)
    else:
        initial_H = np.eye(D)

    value_and_grad = jax.value_and_grad(func)

    f_0, g_0 = value_and_grad(x0)
    state = state._replace(f_k=f_0, g_k=g_0, H_k=initial_H, nfev=state.nfev + 1, ngev=state.ngev + 1,
                           converged=np.linalg.norm(g_0) < g_tol)

    while not state.converged and not state.failed and state.k < maxiter:
        p_k = -np.dot(state.H_k, state.g_k)
        line_search_results = line_search_nojit(value_and_grad, state.x_k, p_k, f_0=state.f_k, g_0=state.g_k,
                                                max_iterations=ls_maxiter)
        state = state._replace(nfev=state.nfev + line_search_results.nfev,
                               ngev=state.ngev + line_search_results.ngev,
                               failed=line_search_results.failed)
        s_k = line_search_results.a_k * p_k
        x_kp1 = state.x_k + s_k
        f_kp1 = line_search_results.f_k
        g_kp1 = line_search_results.g_k
        # print(g_kp1)
        # f_kp1, g_kp1 = value_and_grad(x_kp1)
        # print(g_kp1)
        y_k = g_kp1 - state.g_k
        rho_k = np.reciprocal(np.dot(y_k, s_k))

        sy_k = s_k[:, None] * y_k[None, :]
        w = np.eye(D) - rho_k * sy_k
        H_kp1 = np.dot(np.dot(w, state.H_k), w.T) + rho_k * s_k[:, None] * s_k[None, :]

        converged = np.linalg.norm(g_kp1) < g_tol

        state = state._replace(converged=converged,
                               k=state.k + 1,
                               x_k=x_kp1,
                               f_k=f_kp1,
                               g_k=g_kp1,
                               H_k=H_kp1
                               )
        print(state)
    return state