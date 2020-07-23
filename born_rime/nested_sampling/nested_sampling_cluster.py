import jax.numpy as jnp
from jax.lax import scan, while_loop, dynamic_update_slice, cond, dynamic_slice
from jax import random, vmap
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
    live_points: jnp.ndarray  # [N, M] points in unit cube of live points
    log_L_live: jnp.ndarray  # log likelihood of live points
    dead_points: jnp.ndarray  # [D, M] dead points
    num_dead: int  # int, number of samples (dead points) taken so far.
    log_L_dead: jnp.ndarray  # log likelhood of dead points
    evidence_state: namedtuple  # state for logZ
    cluster_evidence_state: namedtuple  # state for logZp
    m_state: namedtuple  # state for parameter mean
    M_state: namedtuple  # state for parameter covariance
    num_clusters: int  # number of clusters in use
    cluster_centers: jnp.ndarray  # [max_K, M] centers of clusters for k means
    cluster_id: jnp.ndarray  # [N] int, identification of live points to a cluster
    num_per_cluster: jnp.ndarray  # [max_K] int, number of live points in each cluster
    information_gain_state: namedtuple  # information, H, state
    remaining_Z: jnp.ndarray  # estimation of Z left in live points
    running_Z: jnp.ndarray  # mean of accumulated Z
    status: int  # exit status: 0=good, 1=max samples reached


def _masked_cluster_id(points, centers, K):
    max_K = centers.shape[0]
    # max_K, N
    dist = jnp.linalg.norm(centers[:, None, :] - points[None, :, :], axis=-1)
    dist = jnp.where(jnp.arange(max_K)[:, None] > K - 1, jnp.full_like(dist, jnp.inf), dist)
    return jnp.argmin(dist, axis=0)


def _cluster(key, points, max_K=6):
    """
    Cluster `points` adaptively.

    Args:
        key:
        points: [N,M]
        niters:
        max_K:

    Returns: tuple of
        key: PRNG key
        cluster_centers: [max_K, M] (last max_K - K rows are zeros)
        K: int, the number of clusters found.

    """

    # points = points - jnp.mean(points, axis=0)
    # points = points / jnp.maximum(jnp.std(points, axis=0), 1e-8)

    def _init_points(key, points):
        def body(state, X):
            (key, i, center) = state
            key, new_point_key, t_key = random.split(key, 3)
            new_point = points[random.randint(new_point_key, (), 0, points.shape[0]), :]
            dx = points - center
            p = new_point - center
            p = p / jnp.linalg.norm(p)
            t_new = jnp.max(dx @ p, axis=0)
            new_point = center + random.uniform(t_key) * t_new * p
            center = (center * i + new_point) / (i + 1)
            return (key, i + 1, center), (new_point,)

        (key, _, _), (init_points,) = scan(body, (key, 0, points[0, :]), (jnp.arange(max_K),))
        return key, init_points

    def kmeans(key, points, K):
        # key, shuffle_key = random.split(key, 2)
        # centers = random.shuffle(shuffle_key, points, axis=0)
        # centers = centers[:K, :]
        key, centers = _init_points(key, points)
        # N
        cluster_id = _masked_cluster_id(points, centers, K)

        def body(state):
            (done, i, centers, cluster_id) = state

            # [M, max_K]
            new_centers = vmap(lambda coords:
                               jnp.bincount(cluster_id, weights=coords, minlength=max_K, length=max_K))(points.T)
            # max_K, M
            new_centers = new_centers.T
            # max_K
            num_per_cluster = jnp.bincount(cluster_id, minlength=max_K, length=max_K)
            # max_K, M
            new_centers = jnp.where(num_per_cluster[:, None] == 0.,
                                    jnp.zeros_like(new_centers),
                                    new_centers / num_per_cluster[:, None])
            # N
            new_cluster_id = _masked_cluster_id(points, new_centers, K)

            done = jnp.all(new_cluster_id == cluster_id)

            return (done, i + 1, new_centers, new_cluster_id)

        (_, _, centers, _) = while_loop(lambda state: ~state[0],
                                        body,
                                        (jnp.array(False), jnp.array(0), centers, cluster_id))
        return key, centers

    def metric(points, centers, K):
        # N
        cluster_id = _masked_cluster_id(points, centers, K)
        # N,N
        dist = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        # N,N
        in_group = cluster_id[:, None] == cluster_id[None, :]
        in_group_dist, w = jnp.average(dist, weights=in_group, axis=-1, returned=True)
        in_group_dist *= w / (w - 1.)

        # max_K, N, N
        out_group = (~in_group) & (jnp.arange(max_K)[:, None, None] == cluster_id[None, None, :])
        # max_K, N
        out_group_dist = jnp.sum(dist * out_group, axis=-1) / jnp.sum(out_group, axis=-1)
        out_group_dist = jnp.where(jnp.isnan(out_group_dist), jnp.inf, out_group_dist)
        # N
        out_group_dist = jnp.min(out_group_dist, axis=0)
        out_group_dist = jnp.where(jnp.isinf(out_group_dist), jnp.max(in_group_dist), out_group_dist)
        sillohette = (out_group_dist - in_group_dist) / jnp.maximum(in_group_dist, out_group_dist)
        # condition for pos def cov
        sillohette = jnp.where(w < points.shape[1], -jnp.inf, sillohette)
        return jnp.mean(sillohette), cluster_id

    def cluster_probe(K, key):
        key, centers = kmeans(key, points, K)
        sillohette, _ = metric(points, centers, K)
        return centers, sillohette

    key, split_key = random.split(key, 2)
    # test_K = jnp.arange(1, max_K+1)
    # centers_list, sillohettes = vmap(cluster_probe)(test_K, random.split(split_key, max_K))
    # best = jnp.argmax(sillohettes)
    # K = test_K[best]
    # centers = centers_list[best, :, :]

    sillohettes = []
    centers_list = []
    for test_key, K in zip(random.split(split_key, max_K), range(1, max_K + 1)):
        clusters, sillohette = cluster_probe(K, test_key)
        centers_list.append(clusters)
        sillohettes.append(sillohette)
    sillohettes = jnp.stack(sillohettes)
    centers_list = jnp.stack(centers_list, axis=0)
    best = jnp.argmax(sillohettes)
    K = best + 1
    centers = centers_list[best, :, :]

    return key, centers, K, sillohettes


def _recluster(state: NestedSamplerState, max_K: int):
    """
    Cluster the live points into at most max_K clusters.

    Args:
        state:

    Returns:

    """
    # aug_points = jnp.concatenate([state.live_points_U, state.log_L_live[:, None]], axis=1)
    key, cluster_centers, K, _ = _cluster(state.key,
                                          state.live_points,
                                          max_K=max_K)
    # cluster_centers = cluster_centers[:, :-1]

    # initialise clusters
    # K, N
    # dist = jnp.linalg.norm(state.live_points_U[None, :, :] - cluster_centers[:K, None, :], axis=-1)
    # # N
    # cluster_id = jnp.argmin(dist, axis=0)
    cluster_id = _masked_cluster_id(state.live_points, cluster_centers, K)
    # max_K (only first K are non-empty)
    num_per_cluster = jnp.bincount(cluster_id, minlength=max_K, length=max_K)

    # initialise cluster evidence and volume
    cluster_evidence = ClusterEvidence(global_evidence=Evidence(state=state.evidence_state),
                                       num_parent=state.live_points.shape[0],
                                       num_per_cluster=num_per_cluster)
    state = state._replace(key=key,
                           num_clusters=K,
                           cluster_centers=cluster_centers,
                           cluster_id=cluster_id,
                           num_per_cluster=num_per_cluster,
                           cluster_evidence_state=cluster_evidence.state
                           )
    return state


def _expanded_box_restriction(key, log_L_constraint, live_points, cluster_id,
                              spawn_point, spawn_point_cluster_id, cluster_centers,
                              num_per_cluster,
                              loglikelihood_from_U):
    """
    Samples from the prior restricted to the likelihood constraint.
    This undoes the shrinkage at each step to approximate a bound on the contours.
    First it does a scaling on each dimension.

    Args:
        key:
        log_L_constraint:
        live_points:
        cluster_id:
        spawn_point:
        spawn_point_cluster_id:
        num_in_spawn_cluster
        num_repeats:
        loglikelihood_from_U:

    Returns:

    """
    spawn_cluster_center = cluster_centers[spawn_point_cluster_id, :]
    num_in_spawn_cluster = num_per_cluster[spawn_point_cluster_id]
    key, R_key = random.split(key, 2)
    # M,M
    R = random_ortho_matrix(R_key, spawn_point.shape[0])
    weights = cluster_id == spawn_point_cluster_id
    scale = jnp.sqrt(jnp.average(jnp.square(live_points - spawn_cluster_center),
                                 weights=weights, axis=0))

    # initial L, R for each direction
    # t_R[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ R[j,i]
    # t_L[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ -R[j,i]
    # t = ((live_points_U - center)/scale - (spawn_point_U - center)/scale) . R
    # t = (live_points_U - spawn_point_U) . R/scale
    # y_test = (spawn_point_U - center)/scale + U[t_L, t_R].R
    # x_test = scale y_test + center
    # N, M
    dx = (live_points - spawn_point) / scale
    # [N, M]
    t = dx @ R
    t = jnp.where(cluster_id[:, None] == spawn_point_cluster_id, t, 0.)
    # [M]
    t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
    t_L = jnp.minimum(jnp.min(t, axis=0), 0.)

    import pylab as plt
    _live_points = (live_points - spawn_cluster_center)/scale
    _spawn_point = (spawn_point - spawn_cluster_center)/scale
    for i in jnp.unique(cluster_id):
        plt.scatter(_live_points[cluster_id == i, 0], _live_points[cluster_id == i, 1])

    plt.plot([_spawn_point[0], _spawn_point[0] + t_R[0] * R[0,0]], [_spawn_point[1], _spawn_point[1] + t_R[1] * R[1,0]])
    plt.plot([_spawn_point[0], _spawn_point[0] + t_L[0] * R[0,1]], [_spawn_point[1], _spawn_point[1] + t_L[1] * R[1,1]])
    plt.show()

    def body(state):
        (key, i, x_test, log_L_test) = state
        key, uniform_key, beta_key = random.split(key, 3)
        # [M]
        U_scale = random.uniform(uniform_key, shape=spawn_point.shape,
                                 minval=t_L, maxval=t_R)
        t_shrink = random.beta(beta_key, num_in_spawn_cluster, 1)
        u = U_scale / t_shrink
        # y_j =
        #    = dx + sum_i p_i * u_i
        #    = dx + R @ u
        y_test = (spawn_point - spawn_cluster_center)/scale + R @ u
        x_test = y_test * scale + spawn_cluster_center
        log_L_test = loglikelihood_from_U(x_test)
        return (key, i + 1, x_test, log_L_test)

    (key, num_likelihood_evaluations, x_new, log_L_new) = while_loop(lambda state: state[3] <= log_L_constraint,
                                                                     body,
                                                                     (key, 0, spawn_point, log_L_constraint))

    ExpandedBoundResults = namedtuple('ExpandedBoundResults',
                                      ['key', 'num_likelihood_evaluations', 'x_new', 'log_L_new'])
    return ExpandedBoundResults(key, num_likelihood_evaluations, x_new, log_L_new)


def _slice_sampling(key, log_L_constraint, live_points, cluster_id,
                    spawn_point, spawn_point_cluster_id, num_repeats,
                    loglikelihood_from_U):
    """
    Given a spawn point inside the feasible regions, perform a series of
    1D slice samplines.

    Args:
        key: PRNGKey
        spawn_point: [M] point to spawn from.
        live_points: [N, M] all live points.
        cluster_id: [N] int, the id of the live points assigning to a cluster.
        spawn_point_cluster_id: int, cluster if of the spawn point.
        num_repeats: Will cycle through `num_repeats*ndims` random slices.
        loglikelihood_from_U: likelihood_from_U callable

    Returns:

    """

    batched_loglikelihood_from_U = vmap(loglikelihood_from_U)

    def _step_out(x, p, t_L, t_R):
        StepOutState = namedtuple('StepOutState', [
            'bracket', 'num_likelihood_evaluations'])
        w = t_L + t_R
        bracket = jnp.stack([-t_L, t_R])  # 2
        delta = 0.5 * jnp.stack([-w, w])

        # import pylab as plt
        #
        # for i in jnp.unique(cluster_id):
        #     plt.scatter(live_points_U[cluster_id == i, 0], live_points_U[cluster_id == i, 1])
        # plt.plot([spawn_point_U[0], spawn_point_U[0] + t_R * p[0]], [spawn_point_U[1], spawn_point_U[1] + t_R * p[1]])
        # plt.plot([spawn_point_U[0], spawn_point_U[0] - t_L * p[0]], [spawn_point_U[1], spawn_point_U[1] - t_L * p[1]])
        # plt.show()

        def body(state):
            state = state._replace(bracket=state.bracket + delta,
                                   num_likelihood_evaluations=state.num_likelihood_evaluations + 2)
            return state

        def cond(state):
            # 2, M
            check_point = x + p * state.bracket[:, None]
            check_log_likelihood = batched_loglikelihood_from_U(check_point)
            return ~jnp.all(check_log_likelihood <= log_L_constraint)

        state = StepOutState(bracket=bracket, num_likelihood_evaluations=2)
        state = while_loop(cond,
                           body, state)
        # print(state)
        return state

    def _uniformly_sample_1d_slice(key, x, p, L, R):
        Uniform1dSampleState = namedtuple('Uniform1dSampleState', ['key', 'done', 'L', 'R', 'x', 'log_L',
                                                                   'num_likelihood_evaluations'])

        def body(state):
            key, sample_key = random.split(state.key, 2)
            t = state.L + (state.R - state.L) * random.uniform(sample_key)
            x_test = x + t * p
            log_L_test = loglikelihood_from_U(x_test)
            done = log_L_test > log_L_constraint
            L = jnp.where(t < 0., t, state.L)
            R = jnp.where(t > 0., t, state.R)
            state = state._replace(key=key,
                                   done=done,
                                   L=L,
                                   R=R,
                                   log_L=log_L_test,
                                   x=x_test,
                                   num_likelihood_evaluations=state.num_likelihood_evaluations + 1
                                   )
            return state

        state = Uniform1dSampleState(key=key,
                                     done=False,
                                     L=L,
                                     R=R,
                                     x=x,
                                     log_L=log_L_constraint,
                                     num_likelihood_evaluations=0)

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        return state

    OuterSliceSampleState = namedtuple('OuterSliceSampleState', [
        'key', 'x', 'log_L', 'num_likelihood_evaluations'])
    OuterSliceSampleResults = namedtuple('OuterSliceSampleResults', [
        'phantom_x', 'phantom_log_L'])
    InnerSliceSampleState = namedtuple('InnerSliceSampleState', [
        'key', 'i', 'x', 'log_L', 't_R', 't_L', 'num_likelihood_evaluations'
    ])
    outer_state = OuterSliceSampleState(key=key, x=spawn_point, log_L=log_L_constraint,
                                        num_likelihood_evaluations=0)

    def outer_body(outer_state, X):
        key, R_key = random.split(outer_state.key, 2)
        # M,M
        R = random_ortho_matrix(R_key, spawn_point.shape[0])
        # warp so that we explore sampling space

        # print(cholesky, R)
        # Rprime = triangular_solve(cholesky, R, lower=True, transpose_a=True)
        # print(Rprime)

        # initial L, R for each direction
        # t_R[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ R[j,i]
        # t_L[i] = max_(k) (live_points_U[k,j] - spawn_point_U[j]) @ -R[j,i]

        # N, M
        dx = live_points - spawn_point
        # [N, M]
        t = dx @ R
        t = jnp.where(cluster_id[:, None] == spawn_point_cluster_id, t, 0.)
        # [M]
        t_R = jnp.maximum(jnp.max(t, axis=0), 0.)
        t_L = jnp.maximum(jnp.max(-t, axis=0), 0.)

        inner_state = InnerSliceSampleState(
            key=key, i=0, x=outer_state.x, log_L=log_L_constraint, t_L=t_L, t_R=t_R,
            num_likelihood_evaluations=0)

        def inner_body(inner_state):
            """
            Perform series of 1D slice samplings through random O(num_parent, R) basis,
            deformed from the sample space to Unit cube space.
            """
            p = R[:, inner_state.i]
            t_L = inner_state.t_L[inner_state.i]
            t_R = inner_state.t_R[inner_state.i]
            stepout_res = _step_out(inner_state.x, p, t_L, t_R)
            uniform_1d_sample_res = _uniformly_sample_1d_slice(inner_state.key,
                                                               inner_state.x,
                                                               p,
                                                               stepout_res.bracket[0],
                                                               stepout_res.bracket[1])
            inner_state = inner_state._replace(key=uniform_1d_sample_res.key,
                                               i=inner_state.i + 1,
                                               x=uniform_1d_sample_res.x,
                                               log_L=uniform_1d_sample_res.log_L,
                                               num_likelihood_evaluations=inner_state.num_likelihood_evaluations
                                                                          + stepout_res.num_likelihood_evaluations
                                                                          + uniform_1d_sample_res.num_likelihood_evaluations)
            return inner_state

        inner_state = while_loop(lambda state: state.i < R.shape[1],
                                 inner_body,
                                 inner_state)
        outer_state = outer_state._replace(key=inner_state.key,
                                           x=inner_state.x,
                                           log_L=inner_state.log_L,
                                           num_likelihood_evaluations=outer_state.num_likelihood_evaluations
                                                                      + inner_state.num_likelihood_evaluations)
        outer_result = OuterSliceSampleResults(phantom_x=inner_state.x,
                                               phantom_log_L=inner_state.log_L)
        return outer_state, outer_result

    outer_state, outer_result = scan(outer_body, outer_state,
                                     (jnp.arange(num_repeats),))
    # remove last one which is the same as our next sampled point.
    outer_result = outer_result._replace(phantom_x=outer_result.phantom_x[:-1, :],
                                         phantom_log_L=outer_result.phantom_log_L[:-1])
    SliceSampleResult = namedtuple('SliceSampleResult', ['key', 'x', 'log_L', 'phantom_x', 'phantom_log_L',
                                                         'num_likelihood_evaluations'])

    return SliceSampleResult(key=outer_state.key, x=outer_state.x,
                             log_L=outer_state.log_L,
                             phantom_x=outer_result.phantom_x,
                             phantom_log_L=outer_result.phantom_log_L,
                             num_likelihood_evaluations=outer_state.num_likelihood_evaluations)


class NestedSampler(object):
    def __init__(self, loglikelihood, prior_transform: PriorTransform):
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform

        def loglikelihood_from_U(U):
            log_L = loglikelihood(prior_transform(U))
            return jnp.where(jnp.isnan(log_L), -jnp.inf, log_L)

        self.loglikelihood_from_U = loglikelihood_from_U

    def initial_state(self, key, num_live_points, max_samples, max_K):
        # get initial live points
        def single_sample(key):
            U = random.uniform(key, shape=(self.prior_transform.ndims,))
            log_L = self.loglikelihood_from_U(U)
            return U, log_L

        key, init_key = random.split(key, 2)
        live_points, log_L_live = vmap(single_sample)(random.split(init_key, num_live_points))

        dead_points = jnp.zeros((max_samples, self.prior_transform.ndims))
        log_L_dead = jnp.zeros((max_samples,))

        evidence = Evidence()
        m = PosteriorFirstMoment(self.prior_transform)
        M = PosteriorSecondMoment(self.prior_transform)
        cluster_id = jnp.zeros(num_live_points, dtype=jnp.int_)
        num_per_cluster = jnp.bincount(cluster_id, minlength=max_K, length=max_K)
        cluster_evidence = ClusterEvidence(num_parent=num_live_points,
                                           num_per_cluster=num_per_cluster,
                                           global_evidence=evidence)
        information_gain = InformationGain(global_evidence=evidence)
        cluster_centers = jnp.concatenate(
            [jnp.mean(live_points, axis=0, keepdims=True), jnp.zeros((max_K - 1, self.prior_transform.ndims
                                                                      ))], axis=0)

        state = NestedSamplerState(
            key=key,
            done=jnp.array(False),
            i=jnp.array(0),
            num_likelihood_evaluations=num_live_points,
            live_points=live_points,
            log_L_live=log_L_live,
            dead_points=dead_points,
            log_L_dead=log_L_dead,
            num_dead=jnp.array(0),
            evidence_state=evidence.state,
            m_state=m.state,
            M_state=M.state,
            num_clusters=jnp.array(1),
            cluster_evidence_state=cluster_evidence.state,
            cluster_centers=cluster_centers,
            cluster_id=cluster_id,
            num_per_cluster=num_per_cluster,
            information_gain_state=information_gain.state,
            running_Z=jnp.array(0.),
            remaining_Z=jnp.array(1.),
            status=jnp.array(0)
        )

        return state

    def _one_step(self, state: NestedSamplerState, collect_samples: bool):
        # get next dead point
        i_min = jnp.argmin(state.log_L_live)
        dead_point_cluster_id = state.cluster_id[i_min]
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


        n = state.num_per_cluster[dead_point_cluster_id]

        # update tracking
        evidence = Evidence(state=state.evidence_state)
        evidence.update(dead_point, n, log_L_min)
        m = PosteriorFirstMoment(self.prior_transform, state=state.m_state)
        m.update(dead_point, n, log_L_min)
        M = PosteriorSecondMoment(self.prior_transform, state=state.M_state)
        M.update(dead_point, n, log_L_min)
        cluster_evidence = ClusterEvidence(state=state.cluster_evidence_state)
        one_hot_n = jnp.where(jnp.arange(state.cluster_centers.shape[0]) == dead_point_cluster_id,
                              n, 0)
        cluster_evidence.update(dead_point,
                                one_hot_n,
                                log_L_min)
        state = state._replace(evidence_state=evidence.state,
                               m_state=m.state,
                               M_state=M.state,
                               cluster_evidence_state=cluster_evidence.state)

        log_Xp_X = cluster_evidence.X.log_value - evidence.X.log_value
        log_np_n = jnp.log(state.num_per_cluster) - jnp.log(state.live_points.shape[0])

        # select cluster to spawn into
        key, cluster_select_key = random.split(state.key, 2)
        spawn_into_cluster_id = random.categorical(cluster_select_key, log_Xp_X + log_np_n)
        live_points_idx_of_spawn_cluster = jnp.where(state.cluster_id == spawn_into_cluster_id)[0]
        key, spawn_id_key = random.split(key, 2)
        spawn_point_id = live_points_idx_of_spawn_cluster[
            random.randint(spawn_id_key, shape=(), minval=0,
                           maxval=state.num_per_cluster[spawn_into_cluster_id])]
        # slice sample from that point
        # slice_sample_results = _slice_sampling(key,
        #                                             log_L_constraint=log_L_min,
        #                                             spawn_point_U=state.live_points_U[spawn_point_id, :],
        #                                             live_points_U=state.live_points_U,
        #                                             cluster_id=state.cluster_id,
        #                                             spawn_point_cluster_id=state.cluster_id[spawn_point_id],
        #                                             num_repeats=num_repeats,
        #                                        loglikelihood_from_constrained=self.loglikelihood_from_constrained)
        sampler_results = _expanded_box_restriction(key,
                                                    log_L_constraint=log_L_min,
                                                    live_points=state.live_points,
                                                    cluster_id=state.cluster_id,
                                                    spawn_point=state.live_points[spawn_point_id, :],
                                                    spawn_point_cluster_id=state.cluster_id[
                                                        spawn_point_id],
                                                    cluster_centers=state.cluster_centers,
                                                    num_per_cluster=state.num_per_cluster,
                                                    loglikelihood_from_U=self.loglikelihood_from_U)
        # assign new point to a cluster
        # [1]
        new_point_cluster_id = _masked_cluster_id(sampler_results.x_new[None, :],
                                                  state.cluster_centers, state.num_clusters)
        cluster_id = dynamic_update_slice(state.cluster_id, new_point_cluster_id, [i_min])
        num_per_cluster = jnp.bincount(cluster_id,
                                       minlength=state.num_per_cluster.size,
                                       length=state.num_per_cluster.size)
        log_L_live = dynamic_update_slice(state.log_L_live, sampler_results.log_L_new[None], [i_min])
        live_points = dynamic_update_slice(state.live_points, sampler_results.x_new[None, :],
                                           [i_min, 0])

        # Z_live = <L> X_i = exp(logsumexp(log_L_live) - log(N) + log(X))
        remaining_Z = jnp.exp(logsumexp(log_L_live) - jnp.log(log_L_live.size) + evidence.X.log_value)
        running_Z = jnp.exp(evidence.mean)

        state = state._replace(key=sampler_results.key,
                               num_likelihood_evaluations=state.num_likelihood_evaluations +
                                                          sampler_results.num_likelihood_evaluations,
                               cluster_id=cluster_id,
                               num_per_cluster=num_per_cluster,
                               log_L_live=log_L_live,
                               live_points=live_points,
                               running_Z=running_Z,
                               remaining_Z=remaining_Z)

        return state

    def __call__(self, key, num_live_points, max_samples=1e6,
                 recluster_interval=None, max_K=6,
                 collect_samples=True, keep_phantom=True,
                 termination_frac=0.05):
        max_samples = jnp.array(max_samples, dtype=jnp.int64)
        num_live_points = jnp.array(num_live_points, dtype=jnp.int64)
        state = self.initial_state(key, num_live_points,
                                   max_samples=max_samples,
                                   max_K=max_K)

        # for k,a,b in list(zip(state._fields,state, _recluster(state, max_K))):
        #     print(k, a,b)

        if recluster_interval is None:
            recluster_interval = max_samples
        recluster_interval = jnp.array(recluster_interval, dtype=jnp.int64)
        max_K = jnp.array(max_K, dtype=jnp.int64)

        def body(state: NestedSamplerState):
            # sub-clusters and initialises any new clusters.
            state = cond(jnp.equal(jnp.mod(state.i+1, recluster_interval), 0),
                         state, lambda state: _recluster(state, max_K),
                         state, lambda state: state)
            # do one sampling step
            state = self._one_step(state, collect_samples=collect_samples)

            done = state.remaining_Z < termination_frac * state.running_Z
            state = state._replace(done=done,
                                   i=state.i + 1)

            return state

        state = while_loop(lambda state: ~state.done,
                           body,
                           state)
        results = self._finalise_results(state, collect_samples=collect_samples)
        return results

    def _finalise_results(self, state: NestedSamplerState, collect_samples: bool):
        collect = ['logZ',
                   'logZerr',
                   'logZp',
                   'logZperr',
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
        m = PosteriorFirstMoment(self.prior_transform, state=state.m_state)
        M = PosteriorSecondMoment(self.prior_transform, state=state.M_state)
        cluster_evidence = ClusterEvidence(state=state.cluster_evidence_state)
        H = InformationGain(global_evidence=evidence, state=state.information_gain_state)

        data = dict(
            logZ=evidence.mean,
            logZerr=jnp.sqrt(evidence.variance),
            logZp=cluster_evidence.mean,
            logZperr=jnp.sqrt(cluster_evidence.variance),
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
            data['samples'] = state.dead_points
            data['log_L'] = state.log_L_dead
        return NestedSamplerResults(**data)


def debug_nestest_sampler():
    from jax.lax_linalg import triangular_solve
    from jax import random, disable_jit
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

    log_likelihood = lambda x: log_normal(x, data_mu, data_cov)
    prior_transform = PriorTransform(prior_mu, jnp.sqrt(jnp.diag(prior_cov)))
    ns = NestedSampler(log_likelihood, prior_transform)
    with disable_jit():
        results = ns(key=random.PRNGKey(0),
                     num_live_points=100,
                     max_samples=1e6,
                     recluster_interval=None,
                     max_K=6,
                     collect_samples=False,
                     keep_phantom=True,
                     termination_frac=0.05)

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
