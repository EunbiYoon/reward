"""Single-episode LLM judge: is the judge fooled by the personas?

Unlike judge/judge_llm.py (two episodes A and B side by side, for judge preferences), the judge here
reads ONE episode (agent 0 vs its teammate) and gives agent 0's share of the team's result, o(tau).
Nothing else is in the prompt, so o depends on that episode alone and can be compared with its
Shapley share s(tau) from the pool:  gap = o - s  (> 0: the judge over-credits agent 0).

    # judge every trajectory of a pool (resumable: ids already judged are skipped)
    python -m judge.judge_single --run results/judge/need_curve_ppo --out results/judge/0518_1400/need_curve_ppo \
        --judge local --model Qwen/Qwen3.8-27B-FP8 --base-url http://localhost:8000/v1 --prompt basic

    # offline smoke test without a model (credit = own_share, i.e. "who delivered"; never a finding)
    python -m judge.judge_single --run results/judge/need_curve_grpo --judge delivered --limit 6

    # table: per prompt x behaviour, mean o, s, gap, and how often the judge is fooled
    python -m judge.judge_single --out results/judge/0518_1400/need_curve_ppo --report

Prompts (--prompt, text in judge/prompt.py): basic; definition (+ the Shapley definition);
solo (+ definition and this episode's v_partner_alone / v_learner_alone).

Hosted judges (same prompts, instead of Qwen): --judge gemini / --judge openai, key in GEMINI_API_KEY /
OPENAI_API_KEY (read from the project's .env, git-ignored). --rpm caps the request
rate; when the quota runs out the run stops and the same command resumes it later.
--run is the pool (read only: <run>/pool); --out is where results go (default --run), e.g.
results/judge/<timestamp>/<pool>. Writes <out>/single_<judge>_<prompt>.jsonl, one line per trajectory: the judge's answer next to the episode's
events ([[t, {event: count}], ...], suffix 0 = agent 0, 1 = teammate) and events_total. The judge itself reads
every event and every agent's action at every step (judge/describe_llm.py, no cap);
--report writes <out>/single_report.md.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .collect import load_pool, load_traj, make_pool_env
from .common import clip01
from .describe_llm import describe
from .prompt import GAMES, PROMPTS, USER_SINGLE, episode_text, system_prompt_single

SCHEMA = {"type": "object", "required": ["reasoning", "credit"],
          "properties": {"reasoning": {"type": "string"}, "credit": {"type": "number"}}}
FOOLED = 0.1          # gap above this counts as over-credit


class LocalJudge:
    """OpenAI-compatible /chat/completions (vLLM on Unity)."""

    def __init__(self, model: str, base_url: str, think: bool = False, timeout: float = 600):
        self.model, self.url, self.think, self.timeout = model, base_url.rstrip("/"), think, timeout
        try:
            served = json.loads(urllib.request.urlopen(self.url + "/models", timeout=10).read())
        except OSError as e:
            raise SystemExit(f"no OpenAI-compatible server at {self.url} (start vLLM first)") from e
        have = [m["id"] for m in served.get("data", [])]
        if have and model not in have:
            raise SystemExit(f"the server serves {have}, not {model!r}")

    def __call__(self, system: str, user: str) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": 4096 if self.think else 2048,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "credit", "schema": SCHEMA, "strict": True}},
                "chat_template_kwargs": {"enable_thinking": self.think}}
        req = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        resp, c = {}, ""
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            c = resp["choices"][0]["message"]["content"] or ""
            v = json.loads(c[c.find("{"):c.rfind("}") + 1])
            return {"credit": clip01(v["credit"]), "reasoning": v["reasoning"], "served_by": self.model,
                    "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens")}
        except (OSError, KeyError, IndexError, ValueError, TypeError) as e:
            # keep enough to tell a cut-off answer (finish_reason "length") from a malformed one
            ch = (resp.get("choices") or [{}])[0]
            return {"error": f"{type(e).__name__}: {e}", "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                    "raw_head": c[:300], "raw_tail": c[-300:]}


class OllamaJudge:
    """Ollama /api/chat with a JSON schema (e.g. qwen3.8:27b on a Mac)."""

    def __init__(self, model: str, host: str, think: bool = False, timeout: float = 600):
        self.model, self.host, self.think, self.timeout = model, host.rstrip("/"), think, timeout

    def __call__(self, system: str, user: str) -> dict:
        body = {"model": self.model, "stream": False, "format": SCHEMA, "think": self.think,
                "options": {"temperature": 0, "num_ctx": 16384},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        req = urllib.request.Request(self.host + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            v = json.loads(resp["message"]["content"])
            return {"credit": clip01(v["credit"]), "reasoning": v["reasoning"], "served_by": self.model,
                    "prompt_tokens": resp.get("prompt_eval_count")}
        except (OSError, KeyError, ValueError) as e:
            return {"error": f"{type(e).__name__}: {e}"}


API = {   # hosted judges: both speak the OpenAI /chat/completions format
    "gemini": {"url": "https://generativelanguage.googleapis.com/v1beta/openai", "key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
               "model": "gemini-flash-lite-latest", "signup": "https://aistudio.google.com/apikey"},
    "openai": {"url": "https://api.openai.com/v1", "key": ("OPENAI_API_KEY",),
               "model": "gpt-5-mini", "signup": "https://platform.openai.com/api-keys"},
}


def load_dotenv(path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")):
    """KEY=value lines of the project's .env (git-ignored) into os.environ; variables already set win."""
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class RateLimited(Exception):
    pass


class ApiJudge:
    """Gemini or OpenAI over the internet. API key from the environment (never written to any file here).

    rpm caps requests per minute across all workers (free Gemini tiers allow only a few). A 429 is retried
    with backoff; if it persists (e.g. the free daily quota is used up) RateLimited is raised and the run
    stops: rerun the same command (same --out) later and it resumes."""

    def __init__(self, provider: str, model: str | None, rpm: float = 0, reasoning: str | None = None,
                 timeout: float = 300):
        cfg = API[provider]
        self.provider, self.model, self.reasoning, self.timeout = provider, model or cfg["model"], reasoning, timeout
        self.url = os.environ.get(f"{provider.upper()}_BASE_URL", cfg["url"]).rstrip("/")
        load_dotenv()
        self.key = next((os.environ[k] for k in cfg["key"] if os.environ.get(k)), None)
        if not self.key:
            raise SystemExit(f"--judge {provider} needs an API key: export {cfg['key'][0]}=...  ({cfg['signup']})")
        self.gap = 60.0 / rpm if rpm else 0.0
        self.lock, self.next_t = threading.Lock(), 0.0
        self.temperature = True        # dropped automatically for models that only allow the default
        schema = dict(SCHEMA, additionalProperties=False) if provider == "openai" else SCHEMA
        self.fmt = {"type": "json_schema", "json_schema": {"name": "credit", "schema": schema, "strict": True}}

    def _req(self, path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(self.url + path, json.dumps(body).encode() if body is not None else None,
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())

    def models(self) -> list[str]:
        return sorted(m["id"].removeprefix("models/") for m in self._req("/models").get("data", []))

    def _wait_turn(self):
        if not self.gap:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.gap
        time.sleep(max(0.0, t - now))

    def __call__(self, system: str, user: str) -> dict:
        body = {"model": self.model, "max_completion_tokens" if self.provider == "openai" else "max_tokens": 8192,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": self.fmt}
        if self.reasoning:
            body["reasoning_effort"] = self.reasoning
        resp, c = {}, ""
        for attempt in range(6):
            if self.temperature:
                body["temperature"] = 0
            else:
                body.pop("temperature", None)
            self._wait_turn()
            try:
                resp = self._req("/chat/completions", body)
                break
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:500]
                if e.code == 400 and "temperature" in msg and "temperature" in body:
                    self.temperature = False          # e.g. OpenAI reasoning models: default temperature only
                    continue
                if e.code in (429, 500, 502, 503, 504):
                    if attempt == 5:
                        if e.code == 429:
                            raise RateLimited(msg) from e
                        return {"error": f"HTTP {e.code}: {msg}"}
                    time.sleep(float(e.headers.get("Retry-After") or 0) or min(120, 10 * 2 ** attempt))
                    continue
                return {"error": f"HTTP {e.code}: {msg}"}
            except OSError as e:
                if attempt == 5:
                    return {"error": f"{type(e).__name__}: {e}"}
                time.sleep(10 * 2 ** attempt)
        try:
            c = resp["choices"][0]["message"]["content"] or ""
            v = json.loads(c[c.find("{"):c.rfind("}") + 1])
            u = resp.get("usage") or {}
            return {"credit": clip01(v["credit"]), "reasoning": v["reasoning"],
                    "served_by": resp.get("model", self.model), "temperature": 0 if self.temperature else "default",
                    "reasoning_effort": self.reasoning,
                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}
        except (KeyError, IndexError, ValueError, TypeError) as e:
            ch = (resp.get("choices") or [{}])[0]
            return {"error": f"{type(e).__name__}: {e}", "finish_reason": ch.get("finish_reason"),
                    "raw_head": c[:300], "raw_tail": c[-300:]}


def out_path(run: str, judge: str, prompt: str) -> str:
    return os.path.join(run, f"single_{judge}_{prompt}.jsonl")


def persona(behavior: str) -> str:
    """'ckpt:ppo_theft_greedy_s0' -> 'theft'; scripted behaviours stay as they are."""
    if behavior.startswith("ckpt:"):
        parts = behavior[5:].split("_")
        return parts[1] if len(parts) > 1 else behavior
    return behavior


def judge_pool(a):
    pool = []
    for run in a.run:          # one or more pool folders (e.g. theft, pickup, shadow and greedy of one algo)
        pool_dir = os.path.join(run, "pool")
        if not os.path.exists(os.path.join(pool_dir, "meta.jsonl")):
            raise SystemExit(f"{pool_dir}/meta.jsonl not found")
        pool += [{**r, "_run": run} for r in load_pool(pool_dir)]
    ids = [r["id"] for r in pool]
    if len(ids) != len(set(ids)):
        raise SystemExit("the same episode id is in two pools (e.g. two greedy pools): give each only once")
    if a.focus:
        pool = [r for r in pool if a.focus in r["behavior"]]
    pool = sorted(pool, key=lambda r: (r["seed"], r["id"]))
    if a.limit:
        pool = pool[:a.limit]
    env0 = pool[0]["env"]
    env = make_pool_env(env0, pool[0]["layout"], "stay", None, pool[0]["agents"])
    names, stay = env.action_names, env.stay_action
    env.close()

    os.makedirs(a.out, exist_ok=True)
    path = out_path(a.out, a.judge, a.prompt)
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {r["id"] for r in map(json.loads, filter(str.strip, f)) if "error" not in r}
    todo = [r for r in pool if r["id"] not in done]
    print(f"{len(todo)} episodes to judge ({a.judge}, {a.prompt}); {len(done)} already in {path}")

    if a.judge == "local":
        judge = LocalJudge(a.model, a.base_url, a.think)
    elif a.judge == "ollama":
        judge = OllamaJudge(a.model, a.ollama_host, a.think)
    elif a.judge in API:
        judge = ApiJudge(a.judge, a.model, a.rpm, a.reasoning)
        a.model = judge.model
    else:   # offline mock: the "who delivered" rule
        judge = None
    system = system_prompt_single(GAMES[env0], a.prompt, int(pool[0].get("agents", 2)))

    def one(r):
        # the whole episode: totals, EVERY event (no cap) and every agent's action at every step
        text = episode_text(r, describe(r, load_traj(os.path.join(r["_run"], "pool"), r["id"])[1], env0, names, stay,
                                        max_events=10 ** 9), a.prompt)
        user = USER_SINGLE.format(a=text)
        if judge is None:
            out = {"credit": r["own_share"], "reasoning": "mock: own_share", "served_by": None}
        else:
            out = judge(system, user)
        rec = {"pool": r["_run"], "id": r["id"], "behavior": r["behavior"], "persona": persona(r["behavior"]), "seed": r["seed"],
               "judge": a.judge, "prompt": a.prompt, "model": a.model,
               "s": r["s"], "own_share": r["own_share"], "soups": r["soups"], **out}
        if "credit" in rec:
            rec["gap"] = rec["credit"] - r["s"]
        rec["events_total"], rec["events"] = r["events_total"], r["events"]   # what happened, step by step
        return rec

    n_err = 0
    with open(path, "a") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rec = fu.result()
            except RateLimited as e:
                for g in futs:
                    g.cancel()
                ex.shutdown(wait=True, cancel_futures=True)
                raise SystemExit(f"rate limit / quota reached after {i - 1} new judgments ({e}).\n"
                                 f"Saved so far in {path}; rerun the same command later to resume.")
            if "error" in rec:
                n_err += 1
                print(f"  failed {rec['id']}: {rec['error']}")
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] judged ({n_err} failed)", flush=True)
    print(f"done -> {path}")


def corr(x, y) -> float:
    return st.correlation(x, y) if len(x) > 2 and len(set(x)) > 1 and len(set(y)) > 1 else float("nan")


def report(run: str):
    lines = [f"# Single-episode judge: {run}", "",
             "One episode per prompt; o = judge credit for agent 0, s = Shapley share, gap = o - s. "
             f"fooled = share of episodes with gap > {FOOLED}. own = own_share (soups agent 0 delivered).", ""]
    files = sorted(glob.glob(os.path.join(run, "single_*_*.jsonl")))
    if not files:
        raise SystemExit(f"no single_*.jsonl in {run}")
    for path in files:
        recs = [r for r in map(json.loads, filter(str.strip, open(path))) if "credit" in r]
        if not recs:
            continue
        tag = os.path.basename(path)[len("single_"):-len(".jsonl")]
        lines += [f"## {tag}  ({len(recs)} episodes)", "",
                  f"r(o, s) = {corr([r['credit'] for r in recs], [r['s'] for r in recs]):.2f},  "
                  f"r(o, own) = {corr([r['credit'] for r in recs], [r['own_share'] for r in recs]):.2f}", "",
                  "| persona | n | o | s | gap | fooled | own | soups |", "|---|---|---|---|---|---|---|---|"]
        by = defaultdict(list)
        for r in recs:
            by[r["persona"]].append(r)
        for p, rs in sorted(by.items()):
            m = lambda k: st.mean(r[k] for r in rs)
            lines.append(f"| {p} | {len(rs)} | {m('credit'):.2f} | {m('s'):.2f} | {m('gap'):+.2f} | "
                         f"{sum(r['gap'] > FOOLED for r in rs) / len(rs):.2f} | {m('own_share'):.2f} | {m('soups'):.1f} |")
        lines.append("")
    out = os.path.join(run, "single_report.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", nargs="+", help="one or more pool folders (each contains pool/meta.jsonl); read only")
    p.add_argument("--out", help="output dir for single_*.jsonl and the report (default: --run)")
    p.add_argument("--judge", default="local", choices=["local", "ollama", *API, "delivered"],
                   help="local = vLLM, gemini / openai = hosted API (key from GEMINI_API_KEY / OPENAI_API_KEY), "
                        "delivered = offline mock (credit = own_share) for smoke tests")
    p.add_argument("--model", default=None, help="e.g. Qwen/Qwen3.8-27B-FP8; gemini default "
                   f"{API['gemini']['model']}, openai default {API['openai']['model']}")
    p.add_argument("--rpm", type=float, default=0, help="gemini / openai: max requests per minute (0 = no cap)")
    p.add_argument("--reasoning", default=None, help="gemini / openai: reasoning_effort (e.g. none, minimal, low)")
    p.add_argument("--list-models", action="store_true", help="gemini / openai: print the models the key can use")
    p.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_URL", "http://localhost:8000/v1"))
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--think", action="store_true")
    p.add_argument("--prompt", default="basic", choices=PROMPTS)
    p.add_argument("--focus", default=None, help="only episodes whose behaviour contains this text")
    p.add_argument("--limit", type=int, default=0, help="at most this many episodes (0 = the whole pool)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--report", action="store_true", help="only write <run>/single_report.md")
    a = p.parse_args()
    if a.list_models:
        if a.judge not in API:
            raise SystemExit("--list-models needs --judge gemini or --judge openai")
        return print("\n".join(ApiJudge(a.judge, a.model).models()))
    a.out = a.out or (a.run[0] if a.run and len(a.run) == 1 else None)
    if not a.out:
        raise SystemExit("give --run (to judge) and/or --out")
    if a.report:
        return report(a.out)
    if not a.run:
        raise SystemExit("--run (the pool) is needed to judge")
    if a.judge in ("local", "ollama") and not a.model:
        raise SystemExit(f"--judge {a.judge} needs --model")
    judge_pool(a)


if __name__ == "__main__":
    main()
