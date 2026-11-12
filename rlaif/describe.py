"""Turn a recorded trajectory into the text an LLM judge reads.

The judge sees only what an observer of the episode could see: every agent's actions, who picked
up what and who delivered what, step by step. It never sees the counterfactual coalitions behind
Shapley credit (what the team would have done without agent 0). That gap is the point of the
audit: a judge can only credit what looks like work.
"""
from __future__ import annotations

GAMES = {
    "kitchen": (
        "Overcooked (two cooks share a small kitchen). A soup needs 3 onions put into a pot; the pot "
        "then cooks; a cook takes a dish, scoops the soup and carries it to the serving counter. Each "
        "delivered soup is +1 for the TEAM. Cooks can block each other in the narrow kitchen, and items "
        "picked up and put back down on a counter accomplish nothing by themselves."),
    "warehouse": (
        "Multi-robot warehouse. Some shelves are REQUESTED; a robot drives under a shelf, lifts it, "
        "carries it to a goal cell (one delivery, +1 for the TEAM) and later puts it back. Lifting "
        "shelves that are not requested, or lifting and dropping in place, accomplishes nothing."),
    "foraging": (
        "Level-based foraging. Agents and food have levels. Agents next to a food that press LOAD "
        "together lift it if the sum of their levels is at least the food's level; the TEAM's score is "
        "the fraction of all food eaten. Pressing LOAD next to a food that is too heavy accomplishes "
        "nothing unless a teammate joins."),
    "alem": (
        "Alem, a cooperative survival-and-crafting world. Agents gather wood, stone and ore with DO, "
        "place a crafting table, craft pickaxes and swords, drink water and eat cows to stay alive. "
        "Roles differ (warrior, forager, miner) and some trees, ores and animals only yield when 2 or "
        "3 agents act on them together. Each new achievement an agent unlocks (first wood, first "
        "table, first pickaxe, ...) adds points to the TEAM's score. Pressing DO on a resource that "
        "yields nothing, or gathering more than is ever used, accomplishes nothing."),
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
        lines.append(f"  agent {i}: {T - idle} active steps ({idle} idle, {use} {names[-1]}); pickups: {pick_txt}")
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
