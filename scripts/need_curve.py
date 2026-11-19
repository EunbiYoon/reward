"""Overnight pipeline with learning curves: GRPO and PPO steps, strictly one after another.

    .\\venv\\Scripts\\python scripts/need_curve.py                       # the default order below, one run at a time
    .\\venv\\Scripts\\python scripts/need_curve.py --list                # every step it knows
    .\\venv\\Scripts\\python scripts/need_curve.py --steps grpo_A grpo_A_personas   # chosen steps, in that order
    .\\venv\\Scripts\\python scripts/need_curve.py --smoke --root <tmp>  # 6k steps per run, checks the flow

Every run: 2M steps, seeds 0-2, evaluated right after training (--eval), learning curves saved and drawn
(curves/curves.png). Strictly serial by default: one step at a time, and inside a step one seed of one
reward at a time (--workers N runs N at once). Results go to <root>/<algo>/need_curve/<folder>
(root = results).

Steps (default order marked *):
    grpo_A                  A/                  cramped_room, team next to greedy -> the partner A
  * grpo_greedy_personas    greedy_personas/    cramped_room, next to greedy: theft = credit theft,
                                                pickup = fake help, shadow = shadowing
    grpo_A_personas         A_personas/         cramped_room, next to grpo A: team = preparer, theft =
                                                quitting, pickup, shadow (needs grpo_A)
  * grpo_A_<map>            A_<map>/            team next to greedy on asymmetric_advantages, coordination_ring,
                                                counter_circuit_o_1order, forced_coordination (one step per map)
  * ppo_theft               theft/              cramped_room, theft next to greedy (credit theft)
  * ppo_pickup              pickup/             cramped_room, pickup next to greedy (fake help)
  * ppo_shadow              shadow/             cramped_room, shadow next to greedy (shadowing)

Logs: <root>/need_curve_logs/pipeline.log and one log per step. A failed step is logged and the next
step still runs. On Windows the script keeps the PC from going to sleep (the screen may still turn
off). A step's folder is retrained from the start when the step runs again.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OTHER_MAPS = ("asymmetric_advantages", "coordination_ring", "counter_circuit_o_1order", "forced_coordination")
DEFAULT = ["grpo_greedy_personas", *[f"grpo_A_{m}" for m in OTHER_MAPS], "ppo_theft", "ppo_pickup", "ppo_shadow"]


def all_steps():
    """name -> (algo, folder, rewards, partner, layout, needs). partner 'A' = <algo>/need_curve/A seed 0."""
    s = {"grpo_A": ("grpo", "A", ["team"], "greedy", "cramped_room"),
         "grpo_greedy_personas": ("grpo", "greedy_personas", ["theft", "pickup", "shadow"], "greedy",
                                  "cramped_room"),
         "grpo_A_personas": ("grpo", "A_personas", ["team", "theft", "pickup", "shadow"], "A", "cramped_room")}
    for m in OTHER_MAPS:
        s[f"grpo_A_{m}"] = ("grpo", f"A_{m}", ["team"], "greedy", m)
    for name, reward in (("theft", "theft"), ("pickup", "pickup"), ("shadow", "shadow")):
        s[f"ppo_{name}"] = ("ppo", name, [reward], "greedy", "cramped_room")
    return s


def keep_awake(on: bool):
    """Windows: stop the system from sleeping while the script runs (SetThreadExecutionState)."""
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def log(msg: str, path: str):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    steps = all_steps()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", default=DEFAULT, choices=list(steps), metavar="STEP",
                   help="steps to run, in this order (default: " + " ".join(DEFAULT) + ")")
    p.add_argument("--root", default="results", help="results root; folders go to <root>/<algo>/need_curve/")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel runs inside one step (default 1: one seed of one persona at a time)")
    p.add_argument("--smoke", action="store_true", help="6k steps per run: checks the whole flow in minutes")
    p.add_argument("--list", action="store_true", help="print every step and exit")
    a = p.parse_args()
    if a.list:
        for name, (algo, folder, rewards, partner, layout) in steps.items():
            mark = "*" if name in DEFAULT else " "
            print(f"{mark} {name:<38} {algo}/need_curve/{folder:<34} {' '.join(rewards):<30} "
                  f"partner={partner} layout={layout}")
        return

    root = a.root.replace("\\", "/")
    logs = os.path.join(ROOT, root, "need_curve_logs")
    os.makedirs(logs, exist_ok=True)
    main_log = os.path.join(logs, "pipeline.log")
    keep_awake(True)
    t0, status = time.time(), {}
    log(f"steps: {' -> '.join(a.steps)}", main_log)
    try:
        for i, name in enumerate(a.steps, 1):
            algo, folder, rewards, partner, layout = steps[name]
            out = f"{root}/{algo}/need_curve/{folder}"
            if partner == "A":
                partner = f"{root}/{algo}/need_curve/A/models/{algo}_team_greedy_s0.pt"
                if not os.path.exists(os.path.join(ROOT, partner)):
                    log(f"step {i}/{len(a.steps)} {name} skipped: {partner} missing (run {algo}_A first)", main_log)
                    status[name] = False
                    continue
                partner = "ckpt:" + partner
            n_runs = 3 * len(rewards)
            cmd = [sys.executable, "-m", "train.run", "--algos", algo, "--rewards", *rewards,
                   "--partners", partner, "--seeds", "0", "1", "2", "--layout", layout, "--out", out,
                   "--workers", str(min(n_runs, a.workers)), "--eval"] + (["--env-steps", "6000"] if a.smoke else [])
            log(f"step {i}/{len(a.steps)} {name}: {' '.join(cmd[3:])}", main_log)
            t_step = time.time()
            with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8") as f:
                code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            status[name] = code == 0
            log(f"{'done' if code == 0 else 'FAILED'} {name} (exit {code}, {(time.time() - t_step) / 60:.0f} min) "
                f"-> {out}", main_log)
        log(f"all finished in {(time.time() - t0) / 60:.0f} min: "
            + ", ".join(f"{k}={'ok' if v else 'FAILED'}" for k, v in status.items()), main_log)
    finally:
        keep_awake(False)


if __name__ == "__main__":
    main()
