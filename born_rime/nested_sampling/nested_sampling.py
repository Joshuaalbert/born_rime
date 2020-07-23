import jax.numpy as jnp
from jax.lax import scan, while_loop, dynamic_update_slice, cond, dynamic_slice
from jax import random, vmap, tree_multimap
from jax.scipy.special import logsumexp
from jax.lax_linalg import triangular_solve
from typing import NamedTuple
from collections import namedtuple

from born_rime.nested_sampling.prior_transforms import PriorTransform
from born_rime.nested_sampling.param_tracking import Evidence, PosteriorFirstMoment, PosteriorSecondMoment, \
    ClusterEvidence, InformationGain
from born_rime.nested_sampling.utils import random_ortho_matrix


class NestedSamplerState(NamedTuple):
    """
    key: PRNG
    i: Current iteration index
    L_live: The array of likelihood at live points
    L_dead: The array of likelihood at dead points
    live_points_U: The set of live points
    dead_points: The set of dead points
    """
    key: jnp.ndarray
    done: bool
    i: int
    num_likelihood_evaluations: int  # int, number of times logL evaluated.
    live_points_U: jnp.ndarray  # [N, M] points in unit cube of live points
    live_points: jnp.ndarray  # [N, M] points in constrained space of live points
    log_L_live: jnp.ndarray  # log likelihood of live points
    dead_points: jnp.ndarray  # [D, M] dead points
    num_dead: int  # int, number of samples (dead points) taken so far.
    log_L_dead: jnp.ndarray  # log likelhood of dead points
    evidence_state: namedtuple  # state for logZ
    m_state: namedtuple  # state for parameter mean
    M_state: namedtuple  # state for parameter covariance
    information_gain_state: namedtuple  # information, H, state
    status: int  # exit status: 0=good, 1=max samples reached

def _expanded_box_restriction(key, log_L_constraint, live_points_U,
                              spawn_point_U, loglikelihood_from_constrained,
                              prior_transform):
    """
    Samples from the prior restricted to the likelihood constraint.
    This undoes the shrinkage at each step to approximate a bound on the contours.
    First it does a scaling on each dimension.

    Args:
        key:
        log_L_constraint:
        live_points_U:
        spawn_point_U:
        loglikelihood_from_constrained:

    Returns:

    """
    key, R_key = random.split(key, 2)
    # M,M
    R = random_ortho_matrix(R_key, spawn_point_U.shape[0])

    # initial L, R for each direction
    # t_R[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ R[j,i]
    # t_L[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ -R[j,i]
    # t = ((live_points_U - center)/scale - (spawn_point_U - center)/scale) . R
    # t = (live_points_U - spawn_point_U) . R/scale
    # y_test = (spawn_point_U - center)/scale + U[t_L, t_R].R
    # x_test = scale y_test + center
    # N, M
    dx = live_points_U - spawn_point_U
    # [N, M]
    t = dx @ R
    # [M]
    t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
    t_L = jnp.minimum(jnp.min(t, axis=0), 0.)

    # import pylab as plt
    # _live_points = live_points_U
    # _spawn_point = spawn_point_U
    #
    # plt.scatter(_live_points[:, 0], _live_points[:, 1])
    # #x_i = x0_i + R_ij u_j
    # plt.plot([_spawn_point[0] + t_L[0] * R[0,0], _spawn_point[0] + t_R[0] * R[0,0]], [_spawn_point[1] + t_L[0] * R[1,0], _spawn_point[1] + t_R[0] * R[1,0]])
    # plt.plot([_spawn_point[0] + t_L[1] * R[0,1], _spawn_point[0] + t_R[1] * R[0,1]], [_spawn_point[1] + t_L[1] * R[1,1], _spawn_point[1] + t_R[1] * R[1,1]])
    # plt.show()

    def body(state):
        (key, i, u_test, x_test, log_L_test) = state
        key, uniform_key, beta_key = random.split(key, 3)
        # [M]
        U_scale = random.uniform(uniform_key, shape=spawn_point_U.shape,
                                 minval=t_L, maxval=t_R)
        t_shrink = random.beta(beta_key, live_points_U.shape[0], 1)
        u = U_scale / t_shrink
        # y_j =
        #    = dx + sum_i p_i * u_i
        #    = dx + R @ u
        # x_i = x0_i + R_ij u_j
        u_test = spawn_point_U + R @ u
        x_test = prior_transform(u_test)
        log_L_test = loglikelihood_from_constrained(x_test)
        return (key, i + 1, u_test, x_test, log_L_test)

    (key, num_likelihood_evaluations, u_new, x_new, log_L_new) = while_loop(lambda state: state[-1] <= log_L_constraint,
                                                                     body,
                                                                     (key, 0, spawn_point_U, spawn_point_U, log_L_constraint))

    ExpandedBoundResults = namedtuple('ExpandedBoundResults',
                                      ['key', 'num_likelihood_evaluations', 'u_new', 'x_new',  'log_L_new'])
    return ExpandedBoundResults(key, num_likelihood_evaluations, u_new, x_new, log_L_new)


class NestedSampler(object):
    def __init__(self, loglikelihood, prior_transform: PriorTransform):
        def fixed_likelihood(x):
            log_L = loglikelihood(x)
            return jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)

        self.loglikelihood = fixed_likelihood
        self.prior_transform = prior_transform

        def loglikelihood_from_U(U):
            return fixed_likelihood(prior_transform(U))

        self.loglikelihood_from_U = loglikelihood_from_U

    def initial_state(self, key, num_live_points, max_samples):
        # get initial live points
        def single_sample(key):
            U = random.uniform(key, shape=(self.prior_transform.ndims,))
            constrained = self.prior_transform(U)
            log_L = self.loglikelihood(constrained)
            return U, constrained, log_L

        key, init_key = random.split(key, 2)
        live_points_U, live_points, log_L_live = vmap(single_sample)(random.split(init_key, num_live_points))

        dead_points = jnp.zeros((max_samples, self.prior_transform.ndims))
        log_L_dead = jnp.zeros((max_samples,))

        evidence = Evidence()
        m = PosteriorFirstMoment(self.prior_transform)
        M = PosteriorSecondMoment(self.prior_transform)

        information_gain = InformationGain(global_evidence=evidence)

        state = NestedSamplerState(
            key=key,
            done=jnp.array(False),
            i=jnp.array(0),
            num_likelihood_evaluations=num_live_points,
            live_points=live_points,
            live_points_U=live_points_U,
            log_L_live=log_L_live,
            dead_points=dead_points,
            log_L_dead=log_L_dead,
            num_dead=jnp.array(0),
            evidence_state=evidence.state,
            m_state=m.state,
            M_state=M.state,
            information_gain_state=information_gain.state,
            status=jnp.array(0)
        )

        return state

    def _one_step(self, state: NestedSamplerState, collect_samples: bool):
        # get next dead point
        i_min = jnp.argmin(state.log_L_live)
        dead_point = state.live_points[i_min, :]
        log_L_min = state.log_L_live[i_min]
        state = state._replace(
            num_dead=state.num_dead + 1)
        if collect_samples:
            dead_points = dynamic_update_slice(state.dead_points,
                                               dead_point[None, :],
                                               [state.num_dead, 0])
            log_L_dead = dynamic_update_slice(state.log_L_dead,
                                              log_L_min[None],
                                              [state.num_dead])
            state = state._replace(dead_points=dead_points,
                                   log_L_dead=log_L_dead)


        n = state.live_points.shape[0]

        # update tracking
        evidence = Evidence(state=state.evidence_state)
        evidence.update(dead_point, n, log_L_min, from_U=False)
        m = PosteriorFirstMoment(self.prior_transform, state=state.m_state)
        m.update(dead_point, n, log_L_min, from_U=False)
        M = PosteriorSecondMoment(self.prior_transform, state=state.M_state)
        M.update(dead_point, n, log_L_min, from_U=False)
        H = InformationGain(global_evidence=evidence, state=state.information_gain_state)
        H.update(dead_point, n, log_L_min, from_U=False)

        state = state._replace(evidence_state=evidence.state,
                               m_state=m.state,
                               M_state=M.state,
                               information_gain_state=H.state)

        # select cluster to spawn into
        key, spawn_id_key = random.split(state.key, 2)
        spawn_point_id = random.randint(spawn_id_key, shape=(), minval=0,
                           maxval=n)
        sampler_results = _expanded_box_restriction(key,
                                                    log_L_constraint=log_L_min,
                                                    live_points_U=state.live_points_U,
                                                    spawn_point_U=state.live_points_U[spawn_point_id, :],
                                                    loglikelihood_from_constrained=self.loglikelihood,
                                                    prior_transform=self.prior_transform)
        #
        log_L_live = dynamic_update_slice(state.log_L_live, sampler_results.log_L_new[None], [i_min])
        live_points = dynamic_update_slice(state.live_points, sampler_results.x_new[None, :],
                                           [i_min, 0])
        live_points_U = dynamic_update_slice(state.live_points_U, sampler_results.u_new[None, :],
                                           [i_min, 0])

        print(state.i, evidence, m)

        state = state._replace(key=sampler_results.key,
                               num_likelihood_evaluations=state.num_likelihood_evaluations +
                                                          sampler_results.num_likelihood_evaluations,
                               log_L_live=log_L_live,
                               live_points=live_points,
                               live_points_U=live_points_U)

        return state

    def __call__(self, key, num_live_points, max_samples=1e6,
                 collect_samples=True,
                 termination_frac=0.05):
        max_samples = jnp.array(max_samples, dtype=jnp.int64)
        num_live_points = jnp.array(num_live_points, dtype=jnp.int64)
        state = self.initial_state(key, num_live_points,
                                   max_samples=max_samples)

        def body(state: NestedSamplerState):
            # print(list(map(lambda x: type(x), state)))
            # do one sampling step
            state = self._one_step(state, collect_samples=collect_samples)
            evidence = Evidence(state=state.evidence_state)
            # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
            logZ_live = logsumexp(state.log_L_live) - jnp.log(state.live_points.shape[0]) + evidence.X.log_value
            #Z_live < f * Z => logZ_live < log(f) + logZ
            done = logZ_live < jnp.log(termination_frac) + evidence.mean
            state = state._replace(done=done,
                                   i=state.i + 1)
            # print(list(map(lambda x: type(x), state)))
            # exit(0)
            return state

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        results = self._finalise_results(state, collect_samples=collect_samples)
        return results

    def _finalise_results(self, state: NestedSamplerState, collect_samples: bool):
        collect = ['logZ',
                   'logZerr',
                   'ESS',
                   'H',
                   'num_likelihood_evaluations',
                   'efficiency',
                   'param_mean',
                   'param_mean_err',
                   'param_covariance',
                   'param_covariance_err']
        if collect_samples:
            collect.append('samples')
            collect.append('log_L')

        NestedSamplerResults = namedtuple('NestedSamplerResults', collect)
        evidence = Evidence(state=state.evidence_state)
        evidence.update_from_live_points(state.live_points,
                                         state.log_L_live, from_U=False)
        m = PosteriorFirstMoment(self.prior_transform, state=state.m_state)
        m.update_from_live_points(state.live_points,
                                         state.log_L_live, from_U=False)
        M = PosteriorSecondMoment(self.prior_transform, state=state.M_state)
        M.update_from_live_points(state.live_points,
                                         state.log_L_live, from_U=False)
        H = InformationGain(global_evidence=evidence, state=state.information_gain_state)
        H.update_from_live_points(state.live_points,
                                         state.log_L_live, from_U=False)

        data = dict(
            logZ=evidence.mean,
            logZerr=jnp.sqrt(evidence.variance),
            ESS=evidence.effective_sample_size,
            H=H.mean,
            num_likelihood_evaluations=state.num_likelihood_evaluations,
            efficiency=state.num_dead / (
                    state.num_likelihood_evaluations - state.live_points.shape[0]),
            param_mean=m.mean,
            param_mean_err=jnp.sqrt(m.variance),
            param_covariance=M.mean - m.mean[:, None] * m.mean[None, :],
            param_covariance_err=jnp.sqrt(
                M.variance + jnp.sqrt(m.variance[:, None] * m.variance[None, :]))

        )

        if collect_samples:
            data['samples'] = state.dead_points[state.num_dead]
            data['log_L'] = state.log_L_dead
        return NestedSamplerResults(**data)


def debug_nestest_sampler():
    from jax.lax_linalg import triangular_solve
    from jax import random, disable_jit, jit
    def log_normal(x, mean, cov):
        L = jnp.linalg.cholesky(cov)
        dx = x - mean
        dx = triangular_solve(L, dx, lower=True)
        return -0.5 * x.size * jnp.log(2. * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L))) \
               - 0.5 * dx @ dx

    ndims = 2
    prior_mu = 1. * jnp.ones(ndims)
    prior_cov = jnp.diag(jnp.ones(ndims)) ** 2


    data_mu = jnp.zeros(ndims)
    data_cov = jnp.diag(jnp.ones(ndims)) ** 2
    data_cov = jnp.where(data_cov==0., 0.95,data_cov)
    Y = random.multivariate_normal(random.PRNGKey(0), mean=data_mu, cov=data_cov)
    print(Y)
    log_likelihood = lambda x: log_normal(x, Y, jnp.eye(ndims))
    prior_transform = PriorTransform(prior_mu, 10.*jnp.sqrt(jnp.diag(prior_cov)))
    ns = NestedSampler(log_likelihood, prior_transform)
    @jit
    def run():
        return ns(key=random.PRNGKey(0),
           num_live_points=500,
           max_samples=1e6,
           collect_samples=False,
           termination_frac=0.05)
    # with disable_jit():
    results = run()

    print(results)


def debug_cluster(width=1.1):
    import jax.numpy as jnp
    from jax import random, disable_jit
    import numpy as np
    import pylab as plt

    true_num_clusters = 4
    X = jnp.array(np.stack([np.random.normal((i % 4) * 5., width, size=(2,)) for i in range(20)], axis=0))
    with disable_jit():
        key, cluster_centers, K, sillohettes = _cluster(key=random.PRNGKey(1),
                                                        points=X,
                                                        max_K=jnp.array(10))

    cluster_id = _masked_cluster_id(X, cluster_centers, K)
    print(cluster_centers, K, cluster_id)
    plt.plot(list(range(1, len(sillohettes) + 1)), sillohettes)

    plt.show()
    print('Found {} clusters'.format(K))

    for m in range(K):
        plt.scatter(X[m == cluster_id, 0], X[m == cluster_id, 1])
        plt.scatter(cluster_centers[:K, 0], cluster_centers[:K, 1], c='red')
    print(cluster_centers)
    plt.show()

    assert K == true_num_clusters


def debug_masked_cluster_id():
    points = random.normal(random.PRNGKey(0), shape=(100, 3))
    centers = random.normal(random.PRNGKey(0), shape=(10, 3))
    K = 5
    cluster_id = _masked_cluster_id(points, centers, K)
    dist = jnp.linalg.norm(centers[:K, None, :] - points[None, :, :], axis=-1)
    cluster_id_ = jnp.argmin(dist, axis=0)
    assert jnp.all(cluster_id == cluster_id_)


if __name__ == '__main__':
    # debug_masked_cluster_id()
    # debug_cluster()
    debug_nestest_sampler()
#     import pylab as plt
#
#     R = random_ortho_matrix(random.PRNGKey(0), 2)
#     x = random.normal(random.PRNGKey(0), (100, 2))
#     point = x[1, :]
#     dx = x - point
#     t_R = jnp.max(dx @ R, axis=0)
#     t_L = jnp.max(dx @ -R, axis=0)
#     print(t_R, t_L)
#     plt.scatter(x[:, 0], x[:, 1])
#     for m in range(2):
#         plt.plot([point[0], point[0] + R[0, m]], [point[1], point[1] + R[1, m]])
#     plt.show()
#
#     plt.scatter(x[:, 0], x[:, 1])
#     for m in range(2):
#         plt.plot([point[0], point[0] + t_R[m] * R[0, m]], [point[1], point[1] + t_R[m] * R[1, m]])
#         plt.plot([point[0], point[0] - t_L[m] * R[0, m]], [point[1], point[1] - t_L[m] * R[1, m]])
#     plt.figaspect(1.)
#     plt.show()
#
#
#     from jax import jit
#     @jit
#     def get_cluster_choleksy(cluster_idx, cluster_id, live_points_U, num_per_cluster):
#         weights = jnp.where(cluster_id == cluster_idx, 1., 0.)
#         dx = (live_points_U - jnp.sum(weights[:, None] * live_points_U, axis=0)/num_per_cluster[cluster_idx])
#         cov = jnp.sum(weights[:, None, None]*(dx[:, :, None] * dx[:, None, :]), axis=0)/num_per_cluster[cluster_idx]
#         return jnp.linalg.cholesky(cov)
#
#     live_points_U = random.uniform(random.PRNGKey(0), shape=(20, 2))
#     num_per_cluster = jnp.array([10,10])
#
#     cluster_id = jnp.concatenate([0*jnp.ones(10), 1*jnp.ones(10)])
#
#     assert jnp.isclose(get_cluster_choleksy(0, cluster_id, live_points_U, num_per_cluster),
#           jnp.linalg.cholesky(jnp.cov(live_points_U[:10,:], rowvar=False, bias=True))).all()
#     assert jnp.isclose(get_cluster_choleksy(1, cluster_id, live_points_U, num_per_cluster),
#                        jnp.linalg.cholesky(jnp.cov(live_points_U[10:, :], rowvar=False, bias=True))).all()
#
#     print(get_cluster_choleksy(2, cluster_id, live_points_U, num_per_cluster))
#
