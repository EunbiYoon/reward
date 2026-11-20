"""Everything the LLM judge is told: the game rules, the instructions and the two prompt variants.

    GAMES          the rules of each game, as the judge reads them (kitchen = Overcooked-AI)
    SYSTEM         the judging instructions (filled with one GAMES entry)
    DEFINITION     --prompt definition: + a retrieved definition of the Shapley value
    SOLO           --prompt solo: + the definition and each episode's solo results
    USER           the two episodes A and B (texts from judge/describe_llm.py)

Used by judge/judge_llm.py. Changing any text here changes every later judgment: record which version
a judgments_*.jsonl file was made with before comparing runs.
"""
from __future__ import annotations

GAMES = {
    "kitchen": (
        "Overcooked (two cooks share a small kitchen). Each step a cook moves one cell (N, S, E, W), "
        "stays, or INTERACTs with the cell it faces. A cook holds at most one item. Onions come from "
        "onion dispensers and dishes from a dish dispenser; an item can be put down on an empty "
        "counter, where either cook can pick it up. A soup needs 3 onions put into a pot. Once the pot "
        "holds 3 onions, a cook must INTERACT with the pot to start cooking, and the soup then cooks "
        "for 20 steps. A cook holding a dish INTERACTs with the finished pot to scoop the soup, carries "
        "it to the serving counter and INTERACTs to deliver it. Each delivered soup is worth 20 points "
        "to the TEAM (the counts below are in soups). Cooks cannot walk through each other and can "
        "block each other in the narrow kitchen, and items picked up and put back down on a counter "
        "accomplish nothing by themselves."),
    "warehouse": (
        "Multi-robot warehouse (robots share a grid of shelves, aisles and goal cells). Each step a robot "
        "moves FORWARD one cell, turns LEFT or RIGHT, stays, or TOGGLE_LOADs: it lifts the shelf it stands "
        "under or puts its shelf down. An empty robot can drive under shelves, a loaded one cannot. Some "
        "shelves are REQUESTED; carrying a requested shelf onto a goal cell is one delivery, worth 1 to the "
        "TEAM (the counts below are in deliveries), and another shelf becomes requested. Robots cannot "
        "drive through each other and can block each other in the aisles, and lifting a shelf that is not "
        "requested, or lifting and dropping a shelf in place, accomplishes nothing by itself."),
    "foraging": (
        "Level-based foraging (agents collect food items on a grid). Each step an agent moves one cell "
        "(N, S, E, W), stays, or presses LOAD. Agents and food have levels. Agents standing next to a food "
        "that press LOAD at the same step lift it if the sum of their levels is at least the food's level, "
        "and split it in proportion to their levels. The TEAM's score is the fraction of all food on the "
        "field that was lifted (the counts below are in that fraction); the episode ends when no food is "
        "left. Agents cannot share a cell, and pressing LOAD next to a food that is too heavy accomplishes "
        "nothing unless a teammate joins at the same step."),
    "alem": (
        "Alem, a cooperative survival-and-crafting world. Agents gather wood, stone and ore with DO, "
        "place a crafting table, craft pickaxes and swords, drink water and eat cows to stay alive. "
        "Roles differ (warrior, forager, miner) and some trees, ores and animals only yield when 2 or "
        "3 agents act on them together. Each new achievement an agent unlocks (first wood, first "
        "table, first pickaxe, ...) adds points to the TEAM's score. Pressing DO on a resource that "
        "yields nothing, or gathering more than is ever used, accomplishes nothing."),
}

SYSTEM = """You are evaluating teamwork in a cooperative multi-agent game.

{game}

You will read two episodes, A and B, of the same game from the same starting state with the same
teammate(s). Only agent 0 differs between them. Judge which agent 0 was the better teammate: the
one whose behaviour contributed more to what the TEAM achieved. Then estimate, for each episode,
the share of the team's result (0 to 1) that agent 0 deserves credit for. Answer "tie" only if the
two agent 0s contributed equally. Keep the reasoning to a few sentences."""

# --prompt definition: the retrieved definition from the Assignment 7 pilot (Wikipedia, "Shapley value").
DEFINITION = """

Use this definition of contribution. The Shapley value is a method for fairly distributing the total
gain of a cooperative game among the players. A player's Shapley value is its marginal contribution,
the change in the total gain when the player joins a group of other players, averaged over every
order in which the players could join. A player who adds nothing to any group receives nothing, and
players who add the same amount to every group receive the same amount. For two players, a player's
share is the average of (1) the gain it produces on its own and (2) the extra gain it adds when
joining its partner."""

# --prompt solo: the definition plus the counterfactual facts a log cannot show.
SOLO = ("Counterfactual results for this episode (same start, played separately): with agent 0 frozen, "
        "the teammate(s) alone deliver {vp:g}; agent 0 alone, with the teammate(s) frozen, delivers {vl:g}.")

# --prompt without: only the "team without agent 0" counterfactual (no definition, no agent-0-alone number)
WITHOUT = ("Counterfactual result for this episode (same start, played separately): with agent 0 frozen, "
           "the teammate(s) alone deliver {vp:g}.")

# --prompt score: the team-score-only supervisor: no log at all, just the team's result
SCORE = ("Episode length {T} steps, {n} agents. Agent 0 is the agent being judged; {mates}.\n"
         "The team delivered {soups:g} {unit} in total. No other information about the episode is available.")

# More than two agents: the two-player sentence of DEFINITION is replaced, and SOLO / numbers list every
# coalition (the two-player Shapley value needs only v(P) and v(L); three players need all six).
DEFINITION_3 = (" For three players, a player's share is the average of (1) the gain it produces on its own, "
                "(2) the extra gain it adds when joining one other player (averaged over the two partners), and "
                "(3) the extra gain it adds when joining the other two together.")
COALITIONS = ("Counterfactual results for this episode (same start, each played separately with only the "
              "listed agents moving and the others frozen):\n{lines}")


def _mates(n: int) -> str:
    return "agent 1 is its teammate" if n == 2 else \
        "agents " + ", ".join(map(str, range(1, n))) + " are its teammates"


def _coalition_text(r: dict) -> str:
    """All coalition values of a >2-agent pool record (keys v_L<seats> / v_<seats>, eval/coalitions.py)."""
    n, unit = r["agents"], r["unit"]
    names = lambda tag: ", ".join(["agent 0"] * tag.startswith("L") + [f"agent {c}" for c in tag.lstrip("L")])
    rows = [(k[2:], v) for k, v in r.items() if k.startswith("v_") and k not in ("v_partner_alone", "v_learner_alone")]
    rows.sort(key=lambda kv: (len(kv[0].lstrip("L")) + kv[0].startswith("L"), kv[0]))
    lines = [f"  only {names(tag)} moving: {v:g} {unit}" for tag, v in rows]
    return COALITIONS.format(lines="\n".join(lines))

# basic       the episode log
# definition  log + Shapley definition
# solo        log + definition + v(P) and v(L)
# numbers     log + v(P) and v(L), no definition
# without     log + v(P) only, no definition
# score       team score only (no log)
PROMPTS = ("basic", "definition", "solo", "numbers", "without", "score")


def system_prompt(game: str, prompt: str = "basic") -> str:
    return SYSTEM.format(game=game) + (DEFINITION if prompt in ("definition", "solo") else "")


def episode_text(r: dict, text: str, prompt: str = "basic") -> str:
    if prompt == "score":
        return SCORE.format(T=r["T"], n=r["agents"], mates=_mates(r["agents"]), soups=r["soups"], unit=r["unit"])
    if prompt in ("solo", "numbers"):
        if r.get("agents", 2) > 2:
            return text + "\n\n" + _coalition_text(r)
        return text + "\n\n" + SOLO.format(vp=r["v_partner_alone"], vl=r["v_learner_alone"])
    if prompt == "without":
        return text + "\n\n" + WITHOUT.format(vp=r["v_partner_alone"])
    return text


USER = """Episode A
=========
{a}

Episode B
=========
{b}"""

# Single-episode judging (judge/judge_single.py): one episode, no second episode to compare against,
# so the credit depends on that episode alone. Same DEFINITION / SOLO add-ons as the pairwise prompts.
SYSTEM_SINGLE = """You are evaluating teamwork in a cooperative multi-agent game.

{game}

You will read one episode of the game. Judge how much agent 0 contributed to what the TEAM achieved,
compared with its teammate(s). Estimate the share of the team's result (0 to 1) that agent 0 deserves
credit for; the teammate(s) deserve the rest. Keep the reasoning to a few sentences."""

USER_SINGLE = """Episode
=======
{a}"""


def system_prompt_single(game: str, prompt: str = "basic", n_agents: int = 2) -> str:
    if prompt not in ("definition", "solo"):
        return SYSTEM_SINGLE.format(game=game)
    d = DEFINITION if n_agents == 2 else DEFINITION[:DEFINITION.index(" For two players")] + DEFINITION_3
    return SYSTEM_SINGLE.format(game=game) + d
