"""Step 3: a Bradley-Terry reward model fitted to the judge's preferences.

r_theta(o_t, a_t) is an MLP on agent 0's observation and one-hot action; a trajectory's score is
R(tau) = sum_t r_theta(o_t, a_t), and

    P(A preferred to B) = sigmoid(R(A) - R(B))

is fitted to the judge's soft label (its confidence, 0.5 for a tie) with binary cross-entropy.
Validation holds out whole pool seeds, so no episode is seen in both splits. After fitting, the
per-step reward is shifted and scaled so trajectory scores on the pool have mean 0 and std 1,
the scale PPO / GRPO are trained on (environment reward "learned", judge/learned_env.py).

    python -m judge.reward_model --run results/judge/kitchen --judge claude
    python -m judge.reward_model --run results/judge/kitchen --judge shapley     # oracle RM

Writes <run>/rm_<judge>.pt and <run>/rm_<judge>.json (fit and agreement metrics).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from train.deep_policy import mlp

from .collect import load_pool, load_traj
from .common import load_judgments


class RewardNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.f = mlp(obs_dim + n_actions, 1, hidden)
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))
        self.register_buffer("offset", torch.zeros(()))     # per-step shift, set after fitting
        self.register_buffer("scale", torch.ones(()))       # trajectory-score std on the pool

    def raw(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        z = ((obs - self.mean) / self.std).clamp(-10, 10)
        x = torch.cat([z, nn.functional.one_hot(act.long(), self.n_actions).float()], -1)
        return self.f(x).squeeze(-1)

    def forward(self, obs, act):
        """Normalised per-step reward, what the learner is trained on."""
        return (self.raw(obs, act) - self.offset) / self.scale


def save_rm(path: str, net: RewardNet, meta: dict):
    torch.save({"state": net.state_dict(), "meta": meta}, path)


def load_rm(path: str) -> RewardNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    net = RewardNet(m["obs_dim"], m["n_actions"], m["hidden"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


@torch.no_grad()
def score(net: RewardNet, obs: np.ndarray, acts: np.ndarray, normalized: bool = True) -> float:
    o, a = torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(acts)
    return float((net(o, a) if normalized else net.raw(o, a)).sum())


def agent0(obs, acts):
    """(obs, all seats' actions) -> what the reward model reads: obs and agent 0's actions."""
    return obs, acts[:, 0]


def spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 and np.std(rx) and np.std(ry) else float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--judge", default="claude")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16, help="pairs per gradient step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.2, help="fraction of pool seeds held out")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.RandomState(a.seed)

    pool_dir = os.path.join(a.run, "pool")
    pool = {r["id"]: r for r in load_pool(pool_dir)}
    judg = [j for j in load_judgments(a.run, a.judge) if j["A"] in pool and j["B"] in pool]
    if not judg:
        raise SystemExit("no usable judgments")
    trajs = {tid: agent0(*load_traj(pool_dir, tid)) for tid in pool}
    obs_dim = next(iter(trajs.values()))[0].shape[1]
    from .collect import make_pool_env
    r0 = next(iter(pool.values()))
    env = make_pool_env(r0["env"], r0["layout"], "stay", None, r0["agents"])
    n_actions = env.n_actions
    env.close()

    seeds = sorted({r["seed"] for r in pool.values()})
    val_seeds = set(rng.choice(seeds, max(1, int(round(a.val_frac * len(seeds)))), replace=False).tolist())
    split = lambda j: "val" if pool[j["A"]]["seed"] in val_seeds else "train"
    train = [j for j in judg if split(j) == "train"]
    val = [j for j in judg if split(j) == "val"]

    def label(j) -> float:
        """P(A better) implied by the judgment: its confidence (at least 0.5) toward the winner."""
        conf = min(max(float(j.get("confidence", 1.0)), 0.5), 1.0)
        y = 0.5 if j["better"] == "tie" else conf if j["better"] == "A" else 1 - conf
        return a.label_smoothing / 2 + (1 - a.label_smoothing) * y

    net = RewardNet(obs_dim, n_actions, a.hidden)
    all_obs = np.concatenate([trajs[t][0] for t in pool])
    net.mean.copy_(torch.as_tensor(all_obs.mean(0)))
    net.std.copy_(torch.as_tensor(all_obs.std(0) + 1e-3))
    del all_obs
    T = {tid: (torch.as_tensor(o), torch.as_tensor(ac)) for tid, (o, ac) in trajs.items()}
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def R(tid):
        o, ac = T[tid]
        return net.raw(o, ac).sum()

    def evaluate(js):
        if not js:
            return float("nan"), float("nan")
        with torch.no_grad():
            d = torch.stack([R(j["A"]) - R(j["B"]) for j in js])
            y = torch.tensor([label(j) for j in js])
            loss = nn.functional.binary_cross_entropy_with_logits(d, y).item()
            decided = [i for i, j in enumerate(js) if j["better"] != "tie"]
            acc = float(np.mean([(d[i] > 0).item() == (js[i]["better"] == "A") for i in decided])) if decided else float("nan")
        return loss, acc

    best, best_state = float("inf"), None
    for ep in range(a.epochs):
        net.train()
        order = rng.permutation(len(train))
        for i in range(0, len(order), a.batch):
            js = [train[k] for k in order[i:i + a.batch]]
            d = torch.stack([R(j["A"]) - R(j["B"]) for j in js])
            y = torch.tensor([label(j) for j in js])
            loss = nn.functional.binary_cross_entropy_with_logits(d, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        tl, ta = evaluate(train)
        vl, va = evaluate(val)
        key = vl if vl == vl else tl
        if key < best:
            best, best_state = key, {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:>3}: train loss {tl:.3f} acc {ta:.2f} | val loss {vl:.3f} acc {va:.2f}", flush=True)
    net.load_state_dict(best_state)

    # normalise: trajectory scores on the pool get mean 0, std 1
    with torch.no_grad():
        steps = torch.cat([net.raw(*T[t]) for t in pool])
        raw_scores = np.array([R(t).item() for t in pool])
        Tlen = np.array([len(T[t][1]) for t in pool])
        offset = float(steps.mean())
        scale = float(np.std(raw_scores - offset * Tlen)) or 1.0
        net.offset.fill_(offset)
        net.scale.fill_(scale)

    # agreement with the judge and with Shapley ground truth (phi of agent 0)
    ids = list(pool)
    scores = np.array([score(net, *trajs[t]) for t in ids])
    phis = np.array([pool[t]["phi"] for t in ids])
    is_val = np.array([pool[t]["seed"] in val_seeds for t in ids])

    def shapley_acc(js):
        ok = [(R(j["A"]) > R(j["B"])).item() == (pool[j["A"]]["phi"] > pool[j["B"]]["phi"])
              for j in js if abs(pool[j["A"]]["phi"] - pool[j["B"]]["phi"]) > 1e-9]
        return float(np.mean(ok)) if ok else float("nan")

    with torch.no_grad():
        met = {"judge": a.judge, "pairs_train": len(train), "pairs_val": len(val),
               "val_seeds": sorted(val_seeds), "train_loss": evaluate(train)[0], "train_acc_vs_judge": evaluate(train)[1],
               "val_loss": evaluate(val)[0], "val_acc_vs_judge": evaluate(val)[1],
               "val_acc_vs_shapley": shapley_acc(val), "train_acc_vs_shapley": shapley_acc(train),
               "spearman_score_phi_all": spearman(scores, phis),
               "spearman_score_phi_val": spearman(scores[is_val], phis[is_val]),
               "offset": offset, "scale": scale}
    meta = {"obs_dim": obs_dim, "n_actions": n_actions, "hidden": a.hidden, "env": r0["env"],
            "layout": r0["layout"], "agents": r0["agents"], "judge": a.judge, "metrics": met}
    path = os.path.join(a.run, f"rm_{a.judge}.pt")
    save_rm(path, net, meta)
    with open(os.path.join(a.run, f"rm_{a.judge}.json"), "w") as f:
        json.dump(met, f, indent=2)
    print(json.dumps({k: v for k, v in met.items() if k != "val_seeds"}, indent=2))
    print(f"-> {path}\nNext:  python -m train.run --env {r0["env"]} --rewards learned team theft "
          f"--reward-model {path} --out <dir> --eval")


if __name__ == "__main__":
    main()
