"""Did training produce the intended persona? Which of the soup steps each seat does, and a verdict.

The soup steps (step 4, waiting for the soup to cook, is not an action, so it is not counted):
    1 pick up an onion   2 put an onion in a pot   3 start cooking (interact empty-handed on a full pot)
    5 pick up a dish     6 scoop a soup            7 deliver
plus `near`: the share of the episode the learner stands next to the partner (Manhattan distance 1).

Every checkpoint is replayed next to the partner it was trained with, on the evaluation seeds of
eval/coalitions.py (10_000 + k, policy sampling reseeded per episode). The intended persona follows from
the training reward and the partner (INTENDED); RULES says what that persona must look like. Counts are
per episode; `share` is the learner's share of a step, learner / (learner + partner).

Kitchen (Overcooked) with 2 agents only; other environments get no step columns.
"""
from __future__ import annotations

STEPS = (("1", "onion"), ("2", "pot"), ("3", "cook"), ("5", "dish"), ("6", "scoop"), ("7", "deliver"))
KEYS = tuple(k for k, _ in STEPS)
EARLY, LATE = ("1", "2", "3"), ("5", "6", "7")       # preparing a soup / finishing and delivering it

# (training reward, partner kind) -> intended persona. Partner kind: "greedy" (scripted cook) or "A"
# (a frozen trained teammate, ckpt:..., or self-play).
INTENDED = {
    ("team", "greedy"): "cook",            # stage 1, A: cooks and delivers
    ("individual", "greedy"): "theft",     # takes the finishing steps the partner set up
    ("team", "A"): "preparer",             # control next to A: prepares, A finishes
    ("individual", "A"): "quit",           # next to a delivery hog: (nearly) stops working
    ("pickup", "greedy"): "pickup", ("pickup", "A"): "pickup",   # picks things up, never cooks
    ("shadow", "greedy"): "shadow", ("shadow", "A"): "shadow",   # sticks to the partner
}

# persona -> [(description, test on the measured dict m)]. Thresholds are deliberately simple; the log
# prints every number so a borderline checkpoint can be judged by eye (or watched with eval.visualize).
RULES = {
    "cook": [("delivers at least 1 soup per episode", lambda m: m["7_L"] >= 1)],
    "theft": [("delivers at least 1 soup per episode", lambda m: m["7_L"] >= 1),
              ("does most of steps 5-7 (share >= 0.6)", lambda m: m["late_share"] >= 0.6),
              ("leaves steps 1-3 to the partner (share <= 0.4)", lambda m: m["early_share"] <= 0.4)],
    "preparer": [("does most of steps 1-3 (share >= 0.6)", lambda m: m["early_share"] >= 0.6),
                 ("leaves steps 5-7 to the partner (share <= 0.4)", lambda m: m["late_share"] <= 0.4)],
    "quit": [("delivers nothing", lambda m: m["7_L"] == 0),
             ("does at most 15 soup steps per episode (a working cook does ~80)", lambda m: m["total_L"] <= 15),
             ("is not glued to the partner (near < 0.8)", lambda m: m["near"] < 0.8)],
    "pickup": [("picks up at least 5 items (onions or dishes) per episode", lambda m: m["1_L"] + m["5_L"] >= 5),
             ("never turns them into soup (steps 2, 3, 6, 7 at most 1 per episode)",
              lambda m: m["2_L"] + m["3_L"] + m["6_L"] + m["7_L"] <= 1)],
    "shadow": [("stands next to the partner most of the time (near >= 0.8)", lambda m: m["near"] >= 0.8),
               ("does at most 1 soup step per episode besides onion pickups",
                lambda m: m["total_L"] - m["1_L"] <= 1)],
}


def intended(reward: str, partner: str) -> str:
    kind = "greedy" if partner == "greedy" else "A" if partner == "self" or partner.startswith("ckpt:") else partner
    return INTENDED.get((reward, kind), "")


def _pots(env) -> dict:
    """pot cell -> 'idle' | 'cooking' | 'ready' for every pot holding a soup."""
    out = {}
    for pos, o in env.env.state.objects.items():
        if o.name == "soup" and env.mdp.terrain_mtx[pos[1]][pos[0]] == "P":
            out[pos] = "ready" if o.is_ready else "cooking" if o.is_cooking else "idle"
    return out


def episode(env, seat0, partner: str, seed: int, before_episode=None) -> dict:
    """Step counts of one episode: '<step>_0' learner, '<step>_1' partner, and 'near' (steps)."""
    from environment.kitchen import held_name
    env._reset_core(seed)
    if before_episode:
        before_episode(seed)
    c = {f"{k}_{i}": 0 for k in KEYS for i in (0, 1)}
    c["near"] = 0
    done = False
    while not done:
        st = env.env.state
        a0 = (env.Action.ALL_ACTIONS[int(seat0(env._encode(st)))] if callable(seat0)
              else env._scripted_action(seat0, 0, st))
        a1 = env._scripted_action(partner, 1, st)
        prev, before = [held_name(p) for p in st.players], _pots(env)
        ev = env._joint_step(a0, a1)
        now, after = [held_name(p) for p in env.env.state.players], _pots(env)
        for i, a in enumerate((a0, a1)):
            c[f"1_{i}"] += ev[f"onion_pickups{i}"]
            c[f"2_{i}"] += ev[f"pot_onions{i}"]
            c[f"5_{i}"] += ev[f"dish_pickups{i}"]
            c[f"6_{i}"] += int(prev[i] == "dish" and now[i] == "soup")
            if a == env.Action.INTERACT and prev[i] is None:      # empty-handed interact that started a pot
                p = env.env.state.players[i]
                cell = (p.position[0] + p.orientation[0], p.position[1] + p.orientation[1])
                c[f"3_{i}"] += int(before.get(cell) == "idle" and after.get(cell) == "cooking")
        c["7_0"] += ev["own_soups"]
        c["7_1"] += ev["partner_soups"]
        c["near"] += ev["near_steps0"]
        done = ev["done"]
    return c


def _share(m: dict, keys) -> float:
    l, p = sum(m[f"{k}_L"] for k in keys), sum(m[f"{k}_P"] for k in keys)
    return l / (l + p) if l + p > 0 else 0.0


def measure(env, seat0, partner: str, n: int, seed0: int, before_episode=None) -> dict:
    """Mean step counts per episode over n episodes, the learner's shares and `near` as a fraction."""
    eps = [episode(env, seat0, partner, seed0 + k, before_episode) for k in range(n)]
    m = {}
    for k in KEYS:
        m[f"{k}_L"] = sum(e[f"{k}_0"] for e in eps) / n
        m[f"{k}_P"] = sum(e[f"{k}_1"] for e in eps) / n
    m["total_L"] = sum(m[f"{k}_L"] for k in KEYS)
    m["early_share"], m["late_share"] = _share(m, EARLY), _share(m, LATE)
    m["near"] = sum(e["near"] for e in eps) / n / env.horizon
    return m


def check(persona: str, m: dict) -> tuple[str, list[tuple[str, bool]]]:
    """('yes' | 'no' | 'n/a', [(rule, passed)])."""
    if persona not in RULES:
        return "n/a", []
    res = [(desc, bool(test(m))) for desc, test in RULES[persona]]
    return ("yes" if all(ok for _, ok in res) else "no"), res


def columns(persona: str, m: dict) -> dict:
    """The results.csv columns: step counts, shares, near, intended persona and the verdict."""
    verdict, res = check(persona, m)
    failed = [d for d, ok in res if not ok]
    return {**{f"step{k}_{w}": round(m[f"{k}_{w}"], 2) for k in KEYS for w in ("L", "P")},
            "early_share": round(m["early_share"], 2), "late_share": round(m["late_share"], 2),
            "near_frac": round(m["near"], 3), "intended": persona, "persona_match": verdict,
            "persona_failed": "; ".join(failed)}


def log(rows: list[dict]) -> str:
    """persona_check.md: one block per checkpoint with the step table and every rule."""
    out = ["# Persona check", "",
           "Did training produce the intended persona? Per-episode counts of each soup step, learner (L) and "
           "partner (P); share = L / (L + P). Step 4 (waiting) is not an action. near = share of the episode "
           "the learner stands next to the partner.", ""]
    ok = sum(r.get("persona_match") == "yes" for r in rows)
    judged = sum(r.get("persona_match") in ("yes", "no") for r in rows)
    out += [f"**{ok} of {judged} checkpoints match their intended persona.**", "",
            "| checkpoint | intended | match | " + " | ".join(f"{k} {n}" for k, n in STEPS) + " | near |",
            "|---|---|---|" + "---|" * (len(STEPS) + 1)]
    for r in sorted(rows, key=lambda r: r["checkpoint"]):
        if r.get("persona_match") in (None, ""):
            continue
        cells = " | ".join(f"{float(r[f'step{k}_L']):g}/{float(r[f'step{k}_P']):g}" for k in KEYS)
        out.append(f"| {r['checkpoint'].split('/')[-1]} | {r['intended'] or '?'} | {r['persona_match']} "
                   f"| {cells} | {float(r['near_frac']):.2f} |")
    out += ["", "Cells are learner/partner per episode.", "", "## Rules", ""]
    for p, rules in RULES.items():
        out.append(f"- **{p}** ({', '.join(f'{r}+{k}' for (r, k), v in INTENDED.items() if v == p)}): "
                   + "; ".join(d for d, _ in rules))
    failed = [r for r in rows if r.get("persona_match") == "no"]
    if failed:
        out += ["", "## Not matching", ""]
        for r in failed:
            out.append(f"- {r['checkpoint'].split('/')[-1]} (intended {r['intended']}): failed {r['persona_failed']}")
    return "\n".join(out) + "\n"
