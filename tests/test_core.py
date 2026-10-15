"""Unit tests. Run with `pytest` or, without pytest, `python tests/test_core.py`."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rlhack.demos import generate_demos, generate_ranked_trajectories  # noqa: E402
from rlhack.envs import COIN, LAVA, TEST_KINDS, TRUE_WEIGHTS, make_test_env, make_train_env  # noqa: E402
from rlhack.imitation import behavior_cloning  # noqa: E402
from rlhack.irl import bayesian_irl, maxent_irl, trex  # noqa: E402
from rlhack.mdp import (expected_return, greedy_policy, policy_evaluation, state_occupancy,  # noqa: E402
                        value_iteration)
from rlhack.metrics import evaluate_policy  # noqa: E402
from rlhack.method import penalized_reward  # noqa: E402
from rlhack.preferences import SyntheticHuman, pairwise_preferences  # noqa: E402


def test_transitions_are_stochastic_matrices():
    env = make_test_env("lava_shortcut")
    np.testing.assert_allclose(env.P.sum(-1), 1.0)
    assert (env.next_state[env.goal_states] == env.sink).all()


def test_value_iteration_matches_policy_evaluation():
    env = make_test_env("coin_trap")
    R = env.reward(TRUE_WEIGHTS)
    Q, V = value_iteration(env.P, R, env.gamma, tol=1e-10)
    V_pi = policy_evaluation(env.P, R, greedy_policy(Q), env.gamma)
    np.testing.assert_allclose(V, V_pi, atol=1e-6)


def test_occupancy_normalized_and_consistent_with_return():
    env = make_test_env("combined")
    pi = np.full((env.n_states, env.n_actions), 1 / env.n_actions)
    d = state_occupancy(env.P, pi, env.p0, env.gamma)
    assert abs(d.sum() - 1) < 1e-8
    R = env.reward(TRUE_WEIGHTS)
    assert abs(d @ R / (1 - env.gamma) - expected_return(env.P, R, pi, env.p0, env.gamma)) < 1e-8


def test_true_optimum_avoids_lava_and_coin_camping():
    for kind in TEST_KINDS:
        env = make_test_env(kind)
        Q, _ = value_iteration(env.P, env.reward(TRUE_WEIGHTS), env.gamma)
        m = evaluate_policy(env, greedy_policy(Q), TRUE_WEIGHTS)
        assert m["norm_gap"] < 1e-6
        assert m["lava_occ"] < 0.02 and m["coin_occ"] < 0.1


def test_hacked_reward_is_flagged():
    env = make_test_env("coin_trap")
    w_proxy = np.array([-0.02, 1.0, -1.0, 0.5])
    Q, _ = value_iteration(env.P, env.reward(w_proxy), env.gamma)
    m = evaluate_policy(env, greedy_policy(Q), TRUE_WEIGHTS, proxy_reward=env.reward(w_proxy))
    assert m["hacked"] == 1.0 and m["coin_occ"] > 0.5


def _train_setup(n_demos=10, seed=0):
    rng = np.random.default_rng(seed)
    envs = [make_train_env(rng) for _ in range(5)]
    return rng, envs, generate_demos(envs, n_demos, rng, TRUE_WEIGHTS)


def test_train_maps_correlate_coin_with_goal():
    rng = np.random.default_rng(1)
    for _ in range(20):
        env = make_train_env(rng)
        g = env.goal_rc
        for s in env.states_of_type(COIN):
            r, c = env.to_rc(s)
            assert abs(r - g[0]) + abs(c - g[1]) == 1


def test_maxent_learns_goal_positive():
    rng, envs, demos = _train_setup()
    w = maxent_irl(envs, demos, 0.95, 30, iters=60)
    assert w[1] > w[0]  # goal > empty


def test_bayesian_irl_posterior_shapes_and_uncertainty():
    rng, envs, demos = _train_setup(n_demos=5)
    post = bayesian_irl(envs, demos, rng, n_samples=100, burn_in=100)
    assert post.samples.shape == (100, 4)
    assert post.mean[1] > post.mean[0]
    env = make_test_env("lava_shortcut")
    r_pen = penalized_reward(env, post, lam=1.0)
    assert (r_pen <= post.contrast().reward_mean(env) + 1e-12).all()
    assert (penalized_reward(env, post, 1.0, param="raw") <= post.reward_mean(env) + 1e-12).all()
    assert post.reward_std(env)[env.states_of_type(LAVA)].mean() > 0
    c = post.contrast().mean
    assert abs(c[0]) < 1e-9 and abs(c[1] - 1) < 1e-9


def test_scale_reward_task_units():
    from rlhack.utils import scale_reward
    env = make_test_env("lava_shortcut")
    r = scale_reward(env, 7.0 * env.reward(TRUE_WEIGHTS))
    assert np.abs(r).max() <= 1.0 + 1e-12
    goal, empty = env.goal_states[0], env.states_of_type(0)[0]
    assert abs((r[goal] - r[empty]) - 0.5) < 1e-9  # goal-empty contrast -> 1/clip


def test_trex_and_bc_run():
    rng, envs, demos = _train_setup(n_demos=4)
    ranked = generate_ranked_trajectories(envs, 8, rng, TRUE_WEIGHTS)
    prefs = pairwise_preferences(envs, ranked, SyntheticHuman(TRUE_WEIGHTS, rng=rng), rng, max_pairs=20)
    w = trex(prefs)
    assert w.shape == (4,)
    pol = behavior_cloning(envs, demos)
    pi = pol.probs(make_test_env("coin_trap"))
    np.testing.assert_allclose(pi.sum(-1), 1.0)


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} passed")
