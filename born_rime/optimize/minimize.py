from .bfgs_minimize import bfgs_minimize


def minimize(fun, x0, *, method=None, tol=None, options=None):
    """
    Interface to scalar function minimisation.

    This implementation is jittable so long as `fun` is.
    Args:
        fun: jax function
        x0: initial guess, currently only single flat arrays supported.
        method: Available methods: ['BFGS']
        tol: Tolerance for termination. For detailed control, use solver-specific options.
        options: A dictionary of solver options. All methods accept the following generic options:
            maxiter : int
                Maximum number of iterations to perform. Depending on the
                method each iteration may use several function evaluations.

    Returns:

    """
    if method.lower() == 'bfgs':
        return bfgs_minimize(fun, x0, options=options)
    raise ValueError("Method {} not recognised".format(method))


def test_minimize():

    def rosenbrock(x):
        return np.sum(100. * np.diff(x) ** 2 + (1. - x[:-1]) ** 2)


    x0 = np.zeros(2)


    @jax.jit
    def min_op(x0):
        result = minimize(rosenbrock, x0, analytic_initial_hessian=True)
        return result

    jax_res1 = min_op(x0)

    print("Final with analytic initialisation:\n", jax_res1 )

    jax_res1_nojit = minimize_nojit(rosenbrock, x0, analytic_initial_hessian=True)

    assert np.all(jax_res1.x_k == jax_res1_nojit.x_k)

    @jax.jit
    def min_op(x0):
        result = minimize(rosenbrock, x0, analytic_initial_hessian=False)
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
