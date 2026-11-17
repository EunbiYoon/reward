"""Turn a recorded trajectory into the text an LLM judge reads.

The judge sees only what an observer of the episode could see: every agent's actions, who picked
up what and who delivered what, step by step. It never sees the counterfactual coalitions behind
Shapley credit (what the team would have done without agent 0). That gap is the point of the
audit: a judge can only credit what looks like work.
"""
from __future__ import annotations

from .prompt import GAMES  # noqa: F401  (the game rules live in rlaif/prompt.py)

# kitchen recipe steps between the pickups and the delivery (environment/kitchen.py, TRACE_KEYS)
STEP_PHRASE = {
    "pot_onions": "put an onion into a pot",
    "cook_starts": "started a full pot cooking",
    "soup_pickups": "scooped a cooked soup onto a dish",
}

PICKUP_PHRASE = {
    "onion": "picked up an onion", "dish": "picked up a dish",
    "requested": "lifted a REQUESTED shelf", "other": "lifted a shelf that was not requested",
    "lifted": "lifted food (LOAD succeeded)", "failed": "pressed LOAD next to food but nothing was lifted",
    "gathered": "gathered a resource (DO succeeded)", "missed": "pressed DO on a resource but got nothing",
}


def action_string(acts, names, stay: int) -> str:
    """One character per step: '.' idle, '*' the use action (interact / load / toggle), else the
    first letter of the action name. In blocks of 50 so the judge can locate events by time."""
    use = len(names) - 1
    ch = lambda a: "." if a == stay else "*" if a == use else names[a][0]
    s = "".join(ch(int(a)) for a in acts)
    return " ".join(s[i:i + 50] for i in range(0, len(s), 50))


def describe(meta: dict, acts, env_name: str, names, stay: int, max_events: int = 150) -> str:
    """meta: one pool record (rlaif/collect.py). acts: (T, n_agents) action indices."""
    n, T = acts.shape[1], acts.shape[0]
    unit = meta["unit"]
    lines = [f"Episode length {T} steps, {n} agents. Agent 0 is the agent being judged; "
             f"agent{'s' if n > 2 else ''} {', '.join(map(str, range(1, n)))} "
             f"{'are teammates' if n > 2 else 'is its teammate'}.", "", "Totals:"]
    for i in range(n):
        a = acts[:, i]
        idle = int((a == stay).sum())
        use = int((a == len(names) - 1).sum())
        picks = {k: meta["events_total"].get(f"{k}_pickups{i}", 0) for k in meta["pickup_kinds"]}
        pick_txt = ", ".join(f"{v} x {PICKUP_PHRASE.get(k, k)}" for k, v in picks.items() if v) or "none"
        steps = {k: meta["events_total"].get(f"{k}{i}", 0) for k in STEP_PHRASE}
        step_txt = ", ".join(f"{v} x {STEP_PHRASE[k]}" for k, v in steps.items() if v)
        lines.append(f"  agent {i}: {T - idle} active steps ({idle} idle, {use} {names[-1]}); "
                     f"pickups: {pick_txt}" + (f"; recipe steps: {step_txt}" if step_txt else ""))
    lines.append(f"  {unit} delivered by agent 0: {meta['own_soups']:.2f}; "
                 f"by teammate{'s' if n > 2 else ''}: {meta['partner_soups']:.2f}; "
                 f"team total: {meta['soups']:.2f}")

    lines += ["", "Timeline:"]
    raw = []                                        # (t, text), merged below into runs of steps
    for t, ev in meta["events"]:
        for k, v in ev.items():
            if k == "own_soups":
                raw.append((t, f"agent 0 delivered {v:.3g} {unit}"))
            elif k == "partner_soups":
                raw.append((t, f"teammate delivered {v:.3g} {unit}"))
            elif k[:-1] in STEP_PHRASE and k[-1].isdigit():
                raw.append((t, f"agent {k[-1]} {STEP_PHRASE[k[:-1]]}"))
            elif k.endswith(tuple("0123456789")) and "_pickups" in k:
                kind, seat = k.rsplit("_pickups", 1)
                raw.append((t, f"agent {seat} {PICKUP_PHRASE.get(kind, kind)}"))
    runs: list = []                                 # [first t, last t, text, count]
    for t, txt in sorted(raw, key=lambda x: (x[1], x[0])):
        if runs and runs[-1][2] == txt and t == runs[-1][1] + 1:
            runs[-1][1], runs[-1][3] = t, runs[-1][3] + 1
        else:
            runs.append([t, t, txt, 1])
    runs.sort(key=lambda r: r[0])
    ev_lines = [f"  t={a:>3} {txt}" if n == 1 else f"  t={a:>3}-{b} {txt} (every step, {n}x)"
                for a, b, txt, n in runs]
    if len(ev_lines) > max_events:
        ev_lines = ev_lines[:max_events] + [f"  ... ({len(ev_lines) - max_events} more events)"]
    lines += ev_lines or ["  (nothing happened)"]

    legend = ", ".join(f"{nm[0]}={nm}" for i, nm in enumerate(names) if i not in (stay, len(names) - 1))
    lines += ["", f"Actions per step ('.' idle, '*' {names[-1]}, {legend}):"]
    for i in range(n):
        lines.append(f"  agent {i}: {action_string(acts[:, i], names, stay)}")
    return "\n".join(lines)
