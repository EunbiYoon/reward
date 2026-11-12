"""Step 2: LLM-as-a-judge preferences over trajectory pairs (the "AI feedback" of RLAIF).

Pairs are matched: both trajectories share the partner and the episode seed (same start), and
differ only in who sits in seat 0. The judge reads two descriptions (rlaif/describe.py) and says
which agent 0 was the better teammate, how confident it is, and what share of the team's result
it would credit to agent 0 in each. The A/B order is randomised per pair (position bias).

Judges (--judge):
    claude      Claude through the Anthropic SDK (needs `pip install anthropic` and credentials,
                ANTHROPIC_API_KEY or `ant auth login`). Model --model, default claude-opus-5-5.
    openai      ChatGPT through the OpenAI API (OPENAI_API_KEY). Model --model, default gpt-4o.
    ollama      a local open-weights model served by Ollama (no API key, no cost; needs `ollama
                serve` and a pulled model, --model e.g. qwen3.8:27b-mlx). Uses Ollama's
                structured output (JSON schema), so no extra Python packages.
    local       any OpenAI-compatible server (vLLM, SGLang, llama.cpp, Ollama's /v1): a local
                open-weights model on a cluster GPU (scripts/cluster/judge_gpu.sh). --base-url (default
                $LOCAL_LLM_URL or http://localhost:8000/v1) and --model (the served model name).
                JSON-schema constrained output; thinking switched off for Qwen3-style templates
                unless --think. No extra Python packages.
    shapley     oracle: prefers the larger Shapley contribution phi of agent 0. Not AI feedback;
                its reward model is the upper bound the LLM-feedback reward model is compared to.
    activity    OFFLINE MOCK, for smoke tests without an API key: prefers the agent that looks
                busier (pickups, use actions, deliveries). Biased toward busy agents by
                construction, so never report its audit numbers as findings about LLM judges.

    python -m rlaif.judge --run results/rlaif/kitchen --judge claude --pairs 1000 --workers 8
    python -m rlaif.judge --run results/rlaif/kitchen --judge ollama --model qwen3.6:27b-mlx --workers 1
    python -m rlaif.judge --run results/rlaif/kitchen --judge local --model Qwen/Qwen3-32B --workers 32
    python -m rlaif.judge --run results/rlaif/kitchen --judge shapley
    python -m rlaif.judge --run results/rlaif/theft --judge openai --focus ppo_individual --pairs 30

Prompts (--prompt): basic; definition (+ a retrieved definition of the Shapley value); solo
(+ definition and each episode's solo results v_partner_alone / v_learner_alone). --focus keeps
only pairs where one episode's behaviour contains the given text.

Appends to <run>/judgments_<judge>[_<prompt>].jsonl (resumable: pairs already judged are skipped).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from typing import Literal

from .collect import load_pool, load_traj, make_pool_env
from .describe import GAMES, describe

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


def make_pairs(pool: list[dict], n_pairs: int, seed: int = 0) -> list[tuple[dict, dict]]:
    """Every pair of different behaviours on the same (partner, agents, seed), in random A/B
    order, subsampled to n_pairs (0 = all)."""
    rng = random.Random(seed)
    groups: dict = {}
    for r in pool:
        groups.setdefault((r["partner"], r["agents"], r["seed"]), []).append(r)
    pairs = []
    for g in groups.values():
        for x, y in combinations(sorted(g, key=lambda r: r["id"]), 2):
            pairs.append((x, y) if rng.random() < 0.5 else (y, x))
    rng.shuffle(pairs)
    return pairs[:n_pairs] if n_pairs else pairs


def clip01(x) -> float:
    return float(min(1.0, max(0.0, float(x))))


class ShapleyJudge:
    name = "shapley"

    def __call__(self, a: dict, b: dict, text_a: str, text_b: str) -> dict:
        d = a["phi"] - b["phi"]
        better = "tie" if abs(d) < 1e-9 else "A" if d > 0 else "B"
        return {"better": better, "confidence": 1.0, "credit_A": clip01(a["s"]), "credit_B": clip01(b["s"]),
                "reasoning": f"phi_A={a['phi']:.3f} phi_B={b['phi']:.3f}"}


class ActivityJudge:
    """Mock judge: counts what looks like work. See the module docstring."""
    name = "activity"

    @staticmethod
    def busy(r: dict) -> float:
        picks = sum(v for k, v in r["events_total"].items() if "_pickups0" in k)
        return picks + 0.02 * r["use0"] + 0.005 * r["active0"] + 2.0 * r["own_soups"]

    def __call__(self, a, b, text_a, text_b) -> dict:
        ba, bb = self.busy(a), self.busy(b)
        better = "tie" if abs(ba - bb) < 1e-9 else "A" if ba > bb else "B"
        return {"better": better, "confidence": 0.8, "credit_A": ba / (ba + 1), "credit_B": bb / (bb + 1),
                "reasoning": f"busy_A={ba:.2f} busy_B={bb:.2f}"}


class ClaudeJudge:
    prompt = "basic"
    def __init__(self, model: str, effort: str, fallback: bool = True):
        try:
            import anthropic
            from pydantic import BaseModel
        except ImportError as e:
            raise SystemExit("--judge claude needs:  pip install anthropic pydantic") from e
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("--judge claude needs an API key:  export ANTHROPIC_API_KEY=sk-ant-...  "
                             "(create one at https://console.anthropic.com/settings/keys)")
        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model, self.effort, self.fallback = model, effort, fallback
        self.name = "claude"

        class Verdict(BaseModel):
            reasoning: str
            better: Literal["A", "B", "tie"]
            confidence: float           # 0.5 (a guess) .. 1 (certain)
            credit_A: float             # share of the team's result agent 0 deserves in episode A
            credit_B: float
        self.Verdict = Verdict

    def __call__(self, a, b, text_a, text_b, game: str) -> dict:
        kw = dict(model=self.model, max_tokens=16000,
                  system=[{"type": "text", "text": system_prompt(game, self.prompt), "cache_control": {"type": "ephemeral"}}],
                  messages=[{"role": "user", "content": USER.format(a=text_a, b=text_b)}],
                  output_config={"effort": self.effort}, output_format=self.Verdict)
        if self.fallback:     # a declined request is re-run server side on Anthropic's recommended model
            try:
                resp = self.client.beta.messages.parse(**kw, betas=["server-side-fallback-2026-07-01"],
                                                       fallbacks="default")
            except TypeError:  # an SDK without the parameter
                self.fallback = False
                resp = self.client.messages.parse(**kw)
        else:
            resp = self.client.messages.parse(**kw)
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return {"error": f"stop_reason={resp.stop_reason}", "served_by": resp.model}
        v = resp.parsed_output
        return {"better": v.better, "confidence": clip01(v.confidence), "credit_A": clip01(v.credit_A),
                "credit_B": clip01(v.credit_B), "reasoning": v.reasoning, "served_by": resp.model,
                "request_id": resp._request_id}


class OllamaJudge:
    """Same prompt and verdict fields as ClaudeJudge, through Ollama's /api/chat."""
    prompt = "basic"
    SCHEMA = {"type": "object", "required": ["reasoning", "better", "confidence", "credit_A", "credit_B"],
              "properties": {"reasoning": {"type": "string"}, "better": {"type": "string", "enum": ["A", "B", "tie"]},
                             "confidence": {"type": "number"}, "credit_A": {"type": "number"},
                             "credit_B": {"type": "number"}}}

    def __init__(self, model: str, host: str, think: bool = False, timeout: float = 600):
        self.model, self.host, self.think, self.timeout = model, host.rstrip("/"), think, timeout
        self.name = "ollama"
        try:
            tags = json.loads(urllib.request.urlopen(self.host + "/api/tags", timeout=5).read())
        except OSError as e:
            raise SystemExit(f"--judge ollama: no Ollama server at {self.host} (start it with `ollama serve`)") from e
        have = [m["name"] for m in tags.get("models", [])]
        if model not in have and model + ":latest" not in have:
            raise SystemExit(f"--judge ollama: model {model!r} not pulled; have {have} (or `ollama pull {model}`)")

    def __call__(self, a, b, text_a, text_b, game: str) -> dict:
        body = {"model": self.model, "stream": False, "format": self.SCHEMA, "think": self.think,
                "options": {"temperature": 0, "num_ctx": 16384},
                "messages": [{"role": "system", "content": system_prompt(game, self.prompt)},
                             {"role": "user", "content": USER.format(a=text_a, b=text_b)}]}
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            v = json.loads(resp["message"]["content"])
            if v["better"] not in ("A", "B", "tie"):
                raise ValueError(v["better"])
        except (OSError, KeyError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"better": v["better"], "confidence": clip01(v["confidence"]), "credit_A": clip01(v["credit_A"]),
                "credit_B": clip01(v["credit_B"]), "reasoning": v["reasoning"], "served_by": self.model,
                "prompt_tokens": resp.get("prompt_eval_count")}


class LocalJudge(OllamaJudge):
    """Same prompt and verdict fields, through an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, model: str, base_url: str, think: bool = False, timeout: float = 600):
        self.model, self.base_url, self.think, self.timeout = model, base_url.rstrip("/"), think, timeout
        self.name = "local"
        try:
            served = json.loads(urllib.request.urlopen(self.base_url + "/models", timeout=10).read())
        except OSError as e:
            raise SystemExit(f"--judge local: no OpenAI-compatible server at {self.base_url} "
                             f"(start one, e.g. `vllm serve {model}`)") from e
        have = [m["id"] for m in served.get("data", [])]
        if have and model not in have:
            raise SystemExit(f"--judge local: the server serves {have}, not {model!r}")

    def __call__(self, a, b, text_a, text_b, game: str) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": 4096 if self.think else 1024,
                "messages": [{"role": "system", "content": system_prompt(game, self.prompt)},
                             {"role": "user", "content": USER.format(a=text_a, b=text_b)}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "schema": self.SCHEMA, "strict": True}},
                "chat_template_kwargs": {"enable_thinking": self.think}}
        req = urllib.request.Request(self.base_url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            content = resp["choices"][0]["message"]["content"]
            v = json.loads(content[content.find("{"):content.rfind("}") + 1])
            if v["better"] not in ("A", "B", "tie"):
                raise ValueError(v["better"])
        except (OSError, KeyError, IndexError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"better": v["better"], "confidence": clip01(v["confidence"]), "credit_A": clip01(v["credit_A"]),
                "credit_B": clip01(v["credit_B"]), "reasoning": v["reasoning"], "served_by": self.model,
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens")}


class OpenAIJudge(OllamaJudge):
    """Same prompt and verdict fields, through the OpenAI API (needs OPENAI_API_KEY). Default gpt-4o."""

    def __init__(self, model: str, timeout: float = 600):
        self.model, self.timeout, self.think = model, timeout, False
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise SystemExit("--judge openai needs an API key:  export OPENAI_API_KEY=sk-...  "
                             "(create one at https://platform.openai.com/api-keys)")
        self.name = "openai"
        self.schema = {**self.SCHEMA, "additionalProperties": False}

    def __call__(self, a, b, text_a, text_b, game: str) -> dict:
        body = {"model": self.model, "temperature": 0, "seed": 0,
                "messages": [{"role": "system", "content": system_prompt(game, self.prompt)},
                             {"role": "user", "content": USER.format(a=text_a, b=text_b)}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict", "schema": self.schema, "strict": True}}}
        req = urllib.request.Request(self.base_url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            v = json.loads(resp["choices"][0]["message"]["content"])
            if v["better"] not in ("A", "B", "tie"):
                raise ValueError(v["better"])
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"}
        except (OSError, KeyError, IndexError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}
        return {"better": v["better"], "confidence": clip01(v["confidence"]), "credit_A": clip01(v["credit_A"]),
                "credit_B": clip01(v["credit_B"]), "reasoning": v["reasoning"], "served_by": resp.get("model"),
                "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens")}


def judge_path(run: str, judge: str) -> str:
    return os.path.join(run, f"judgments_{judge}.jsonl")


def load_judgments(run: str, judge: str) -> list[dict]:
    path = judge_path(run, judge)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run  python -m rlaif.judge --run {run} --judge {judge}")
    with open(path) as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if "error" not in r]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the --out of rlaif.collect")
    p.add_argument("--judge", default="claude", choices=["claude", "openai", "ollama", "local", "shapley", "activity"])
    p.add_argument("--model", default=None, help="claude: default claude-opus-5-5; ollama: required, e.g. qwen3.6:27b-mlx")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1"),
                   help="local: the OpenAI-compatible endpoint")
    p.add_argument("--think", action="store_true", help="ollama / local: let a thinking model think (slower)")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--no-fallback", action="store_true", help="do not send fallbacks='default'")
    p.add_argument("--pairs", type=int, default=0, help="number of pairs to judge (0 = all matched pairs)")
    p.add_argument("--workers", type=int, default=8, help="concurrent API requests")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--focus", default=None,
                   help="only pairs where one episode's behaviour contains this text (e.g. ppo_individual)")
    p.add_argument("--prompt", default="basic", choices=PROMPTS,
                   help="basic; definition = + Shapley definition; solo = + definition and each episode's "
                        "solo results. Non-basic prompts write judgments_<judge>_<prompt>.jsonl")
    a = p.parse_args()
    llm = a.judge in ("claude", "openai", "ollama", "local")
    if a.judge == "claude":
        a.model = a.model or "claude-opus-5-5"
    elif a.judge == "openai":
        a.model = a.model or "gpt-4o"
    elif a.judge in ("ollama", "local") and not a.model:
        raise SystemExit(f"--judge {a.judge} needs --model (the served model name)")
    if a.ollama_host and not a.ollama_host.startswith("http"):
        a.ollama_host = "http://" + a.ollama_host

    pool_dir = os.path.join(a.run, "pool")
    pool = load_pool(pool_dir)
    env0 = pool[0]["env"]
    env = make_pool_env(env0, pool[0]["layout"], "stay", None, pool[0]["agents"])
    names, stay = env.action_names, env.stay_action
    env.close()

    judge = (ClaudeJudge(a.model, a.effort, not a.no_fallback) if a.judge == "claude"
             else OpenAIJudge(a.model) if a.judge == "openai"
             else OllamaJudge(a.model, a.ollama_host, a.think) if a.judge == "ollama"
             else LocalJudge(a.model, a.base_url, a.think) if a.judge == "local"
             else ShapleyJudge() if a.judge == "shapley" else ActivityJudge())
    judge.prompt = a.prompt
    tag = a.judge if a.prompt == "basic" or not llm else f"{a.judge}_{a.prompt}"
    path = judge_path(a.run, tag)
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {r["pair"] for r in (json.loads(line) for line in f if line.strip()) if "error" not in r}
    cand = make_pairs(pool, 0, a.seed)
    if a.focus:
        cand = [(x, y) for x, y in cand if a.focus in x["behavior"] or a.focus in y["behavior"]]
    cand = cand[:a.pairs] if a.pairs else cand
    pairs = [(x, y) for x, y in cand if f"{x['id']}|{y['id']}" not in done]
    print(f"{len(pairs)} pairs to judge with {a.judge}"
          + (f" ({a.model}, effort={a.effort})" if a.judge == "claude" else f" ({a.model})" if llm else "")
          + (f"; {len(done)} already in {path}" if done else ""))

    text_cache: dict = {}

    def text(r):
        if r["id"] not in text_cache:
            text_cache[r["id"]] = describe(r, load_traj(pool_dir, r["id"])[1], env0, names, stay)
        return text_cache[r["id"]]

    def one(pair):
        x, y = pair
        args = (x, y, episode_text(x, text(x), a.prompt), episode_text(y, text(y), a.prompt))
        out = judge(*args, GAMES[env0]) if llm else judge(*args)
        return {"pair": f"{x['id']}|{y['id']}", "A": x["id"], "B": y["id"], "judge": tag, "prompt": a.prompt,
                "model": a.model if llm else None, **out}

    n_err = 0
    with open(path, "a") as f:
        if llm:
            api_errors = (judge.anthropic.APIStatusError, judge.anthropic.APIConnectionError) \
                if a.judge == "claude" else ()
            with ThreadPoolExecutor(a.workers) as ex:
                futs = [ex.submit(one, pr) for pr in pairs]
                for i, fu in enumerate(as_completed(futs), 1):
                    try:
                        rec = fu.result()
                    except api_errors as e:
                        n_err += 1
                        print(f"  API error: {getattr(e, 'message', e)}")
                        continue
                    if "error" in rec:
                        n_err += 1
                        print(f"  failed: {rec['error']}")
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if i % 25 == 0 or i == len(pairs):
                        print(f"  [{i}/{len(pairs)}] judged ({n_err} failed)", flush=True)
        else:
            for pr in pairs:
                f.write(json.dumps(one(pr)) + "\n")
    print(f"done -> {path}. Next:  python -m rlaif.reward_model --run {a.run} --judge {tag}")


if __name__ == "__main__":
    main()
