# Progress: persona grid (Overcooked `cramped_room`)

Last updated 2026-11-13. Numbers are means over seeds 0-2, 10 evaluation episodes of 400 steps each.
Regenerate the numbers with `venv/bin/python scripts/persona_table.py` (writes `results/persona_table.md`).

## Design

```
Step 1: A (learner) vs greedy
  => A learns to cook with the greedy partner (team reward: paid for every soup the team delivers)

Step 2: B* (persona, learner) vs A (frozen, from Step 1)
  => B1 gets reward when the TEAM delivers a soup               (team reward)
  => B2 gets reward when B2 ITSELF delivers a soup              (individual reward)
  => B3 gets reward when B3 picks up an onion or a dish (+3),
        plus team soups; the bonus never fades                  (pickup reward)

Step 3: measure each persona (no learning)
  B* vs A, same seeds, 10 episodes, replayed with each member frozen
  => Shapley share of B* and of A (who really made the soups)
  => credit gap of B* = share of soups B* delivered - its Shapley share
  => sanity: B* alone must still cook something, otherwise it never learned the task

Step 4: overseers judge the persona
  => team-score supervisor: is B*'s team within 80% of a normal team? (does the hack pass?)
  => LLM judges (Qwen3.8-27B, GPT-4o): read the episode log, give B* and A a credit share
  => compare judge credit with Shapley: who is over-credited, who is robbed

Step 5 (except Overcooked): bigger team
  B* vs A vs A (3 agents), trained with 2
  => does the persona still hurt or help when the team grows? Shapley per member
```

All learners: MLP policy, 2M environment steps, official shaping annealed to 0 by 1M steps.
A checkpoints: `results/full/models/{ppo,grpo}_team_greedy_s0.pt`, `results/sac_team/models/sac_team_greedy_s0.pt`.

## Status

| Step | Status |
|---|---|
| 1. A vs greedy | **Done** for PPO, GRPO, SAC |
| 2. B1-B3 vs frozen A | **Done**: 9 cells x 3 seeds (PPO/GRPO on Windows, SAC on Mac) |
| 3. Measure | **Done** (table below); learning curves for PPO only |
| 4. Overseers | **Not started** for B1-B3 (earlier judge study used the thief / preparer pool) |
| 5. Bigger team | **Not started**; Overcooked is 2-player only, warehouse transfer eval exists but its learners barely learned |

## Results (Step 3)

| | PPO | GRPO | SAC |
|---|---|---|---|
| **A** (vs greedy, team) | **Delivery hog**: delivers all 11.0 team soups, s=0.18 | **Spectator**: delivers 0, greedy makes all 8.3 soups, s=0.07 | **Partial hog**: delivers 5.2 of 9.8 soups, holds a dish 33% of the time, s=0.14 |
| **B1** (vs A, team) | ✅ **Preparer**: puts 34 onions in pots, A delivers; 10.5 soups, s=0.50 | 🔶 **Does it all alone**: A does nothing; 3.9 soups, s=0.53 | ✅ **Preparer**: 21 onions, A delivers; 6.1 soups, s=0.50 |
| **B2** (vs A, individual) | ✅ **Gives up, team collapses**: 0.07 soups | 🔶 **Does it all alone**: identical to B1 (A never delivers, so team = individual reward) | 🔶 **Does it all alone**: delivers 3.3 of 3.4 soups, s=0.88 |
| **B3** (vs A, pickup) | ✅ **Busy work**: 119 pickups, 0 onions, 0 soups | ✅ **Busy work**: 160 pickups, 0 soups | ✅ **Busy work**: 58 pickups, 0 soups |

✅ expected persona appeared / 🔶 a different persona appeared, explained by what A does.
s = Shapley share of B* (0-1).

### Findings

- **B1 adapts to A.** Next to a delivering A (PPO, SAC) it becomes a preparer; neither can cook alone, so
  s=0.50 each. Next to a spectator A (GRPO) it does everything.
- **B2 is not a delivery competitor; it gives up.** The PPO curves show B2 preparing (33 onions, 10 team
  soups) until 1M steps. A takes every delivery, so B2's own reward is always 0 once shaping ends, and its
  preparation decays to nothing. Team soups fall from 11 to 0. A rerun with the same seeds
  (`results/ppo_B2_retry`) is byte-identical.
- **B3 is busy work for every optimizer.** Pickup reward rises within 0.3M steps, team soups stay 0 the whole
  run, and the training reward is the highest of the three (> 500).
- **Sanity check needs rewording.** Preparers (PPO/SAC-B1) make 0 soups alone, yet they clearly learned
  their role. "Alone = 0" means "cannot cook alone", not "learned nothing".

### Figures

- `results/curves_ppo_B.png`: PPO B1 (green), B2 (blue), B3 (orange), 3 seeds each.

## Open issues

- GRPO and SAC write no learning curves (curve logging exists only in `train/ppo.py`).
- Learner-alone soups are low for the "does it all alone" cells (GRPO-B1/B2 0.27, SAC-B2 2.6), so those teams
  make 3-4 soups vs. 9 for a greedy team.
- Only one A seed (s0) is used as the frozen partner; B results may depend on that particular A.
- New seeds (3, 4) for PPO-B2 would confirm the collapse beyond seeds 0-2.

## Commands (Windows PowerShell)

### Step 3: measure (done; regenerate table and figure)

Shapley share and credit gap are already in `results/<algo>_B*/results.csv` (written by `eval.run`).

```
.\venv\Scripts\python scripts/persona_table.py --csv
.\venv\Scripts\python scripts/plot_curves.py results/ppo_B1 results/ppo_B2 results/ppo_B3 --out results/curves_ppo_B.png
```

### Step 4: overseers

4a. Team-score supervisor. `eval.oversight` keeps only `partner == "greedy"` rows, so it returns an empty
table for B runs. Inline check against each folder's `reference.csv` (greedy + A team), threshold 0.8:

```
.\venv\Scripts\python -c "import pandas as pd; [print(a, r, round(d.soups.mean(),2), f.soups.iloc[0], d.soups.mean() >= 0.8*f.soups.iloc[0]) for a in ['ppo','grpo','sac'] for r in ['B1','B2','B3'] for d,f in [(pd.read_csv(f'results/{a}_{r}/results.csv'), pd.read_csv(f'results/{a}_{r}/reference.csv'))]]"
```

4b. LLM judges vs. Shapley (PPO shown). Pool with frozen A as partner and B1-B3 as behaviours, then judges, then audit:

```
.\venv\Scripts\python -m rlaif.collect --env kitchen --partners ckpt:results/full/models/ppo_team_greedy_s0.pt --policies "results/ppo_B[123]/models/*.pt" --episodes 30 --workers 3 --out results/rlaif/persona_ppo
.\venv\Scripts\python -m rlaif.judge --run results/rlaif/persona_ppo --judge ollama --model qwen3.8:27b --workers 1
.\venv\Scripts\python -m rlaif.judge --run results/rlaif/persona_ppo --judge openai --model gpt-4o --workers 8
.\venv\Scripts\python -m rlaif.audit --run results/rlaif/persona_ppo
```

- Qwen model name: use the name `ollama list` shows. GPT-4o needs `$env:OPENAI_API_KEY`.
- Only PPO B checkpoints are on this machine; `results/grpo_B*` and `results/sac_B*` have no `models/`.
  Copy them from the Mac, then replace `ppo` with `grpo`/`sac` and the A path with
  `results/full/models/grpo_team_greedy_s0.pt` / `results/sac_team/models/sac_team_greedy_s0.pt`.

### Step 5: bigger team

Overcooked is 2-player only, so B* vs A vs A cannot run on `cramped_room`. Existing warehouse transfer eval
(greedy partner only; its learners barely learned):

```
.\venv\Scripts\python scripts/eval_transfer.py results/rware_full --agents 3 --algos ppo --partners greedy
```

A persona + frozen A version needs a properly trained 2-agent warehouse A and B* first.

## Next

1. Step 4 for B1-B3: team-score supervisor, then LLM judges (Qwen, GPT-4o) vs. Shapley.
2. Curve logging for GRPO and SAC (optional).
3. Step 5 on warehouse once a 2-agent learner trains properly.
