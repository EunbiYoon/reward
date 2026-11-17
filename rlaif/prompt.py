"""Everything the LLM judge is told: the game rules, the instructions and the two prompt variants.

    GAMES          the rules of each game, as the judge reads them (kitchen = Overcooked-AI)
    SYSTEM         the judging instructions (filled with one GAMES entry)
    DEFINITION     --prompt definition: + a retrieved definition of the Shapley value
    SOLO           --prompt solo: + the definition and each episode's solo results
    USER           the two episodes A and B (texts from rlaif/describe_llm.py)

Used by rlaif/judge_llm.py. Changing any text here changes every later judgment: record which version
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

PROMPTS = ("basic", "definition", "solo")


def system_prompt(game: str, prompt: str = "basic") -> str:
    return SYSTEM.format(game=game) + (DEFINITION if prompt in ("definition", "solo") else "")


def episode_text(r: dict, text: str, prompt: str = "basic") -> str:
    if prompt != "solo":
        return text
    return text + "\n\n" + SOLO.format(vp=r["v_partner_alone"], vl=r["v_learner_alone"])


USER = """Episode A
=========
{a}

Episode B
=========
{b}"""
