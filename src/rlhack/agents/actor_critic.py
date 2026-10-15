"""One-Step Actor-Critic (episodic), Sutton & Barto (2nd ed.), Section 13.5.

Tabular softmax actor, TD(0) state-value critic, trained from scratch by
sampling the environment.

``use_I=True`` gives the textbook update with the gamma^t factor I; we default to
False (the common practical variant) because with I the actor barely learns from
late-episode states and the oracle agent gets stuck in local optima on our maps. The agent only ever sees the reward vector it is given
(a learned / penalized proxy), never the true reward.
"""
from __future__ import annotations

import numpy as np

from ..utils import sample_categorical, softmax


class OneStepActorCritic:
    def __init__(self, n_states: int, n_actions: int, gamma: float, alpha_theta: float = 1.0,
                 alpha_w: float = 0.5, use_I: bool = False):
        self.theta = np.zeros((n_states, n_actions))
        self.v = np.zeros(n_states)
        self.gamma, self.alpha_theta, self.alpha_w, self.use_I = gamma, alpha_theta, alpha_w, use_I

    def policy(self) -> np.ndarray:
        return softmax(self.theta, axis=-1)

    def train(self, env, reward: np.ndarray, rng, episodes: int = 4000, horizon: int = 40,
              exploring_starts: float = 0.8) -> np.ndarray:
        """Returns per-episode discounted *proxy* return (for learning curves).

        ``exploring_starts`` is the fraction of training episodes that start from a
        uniformly random non-sink state (Sutton & Barto's exploring-starts trick);
        the rest start from env.start. Evaluation always uses env.start.
        """
        curve = np.zeros(episodes)
        n_cells = env.n_states - 1
        for ep in range(episodes):
            s = int(rng.integers(n_cells)) if rng.random() < exploring_starts else env.start
            I, G = 1.0, 0.0
            for t in range(horizon):
                pi_s = softmax(self.theta[s])
                a = sample_categorical(pi_s, rng)
                s2 = env.sample_next(s, a, rng)
                r = reward[s]
                G += self.gamma**t * r
                delta = r + self.gamma * self.v[s2] - self.v[s]
                self.v[s] += self.alpha_w * delta
                grad_log = -pi_s
                grad_log[a] += 1.0
                self.theta[s] += self.alpha_theta * I * delta * grad_log
                if self.use_I:
                    I *= self.gamma
                s = s2
                if s == env.sink:
                    break
            curve[ep] = G
        return curve


def train_actor_critic(env, reward, rng, **kw):
    ac_kw = {k: kw.pop(k) for k in ("alpha_theta", "alpha_w", "use_I") if k in kw}
    agent = OneStepActorCritic(env.n_states, env.n_actions, env.gamma, **ac_kw)
    curve = agent.train(env, reward, rng, **kw)
    return agent.policy(), curve
