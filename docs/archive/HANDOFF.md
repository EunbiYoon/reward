# Handoff: AAAI-27 workshop paper (credit hacking), continue on Windows

Paste this file into the new conversation. Written 2026-11-11.
The CS 690S check-in (`report/checkin.tex`) is being finished separately on the Mac; this session is
only the AAAI paper.

## Target
- Venue: AAAI-27 workshops, Montreal, Feb 22-23, 2027. **Deadline 2026-11-20** (9 days).
- Best fit: **WMAC 2027: Measuring Collective Intelligence in Agentic Systems** (https://multiagents.org),
  deadline 2026-11-20 23:59 UTC, full paper up to 8 pages / short up to 4, non-archival.
  Verify the deadline and format on the site (it did not load when checked).
- Second choice: Effective and Efficient Evaluation for AI Scientists
  (https://eval-ai-scientists.github.io/aaai27/), deadline 2026-11-20 06:00 UTC, 6-8 pages,
  double-blind; fits the LLM-as-judge part.
- **All AAAI-27 workshops require in-person presentation by an author (no virtual).** The author
  graduates in Dec 2026 and will be on OPT: travel to Canada needs an EAD card in hand, an I-20 travel
  signature, and DSO (UMass ISSO) confirmation. Alternative: a co-author presents.

## Story
A reward is also a credit rule, and every overseer that sees outcomes rather than counterfactuals
misses credit theft: (1) RL learners trained with common rewards fake help or steal credit in
Overcooked, measured by the gap between reward credit and Shapley contribution; (2) a team-score
supervisor does not notice (9/13 theft runs look normal); (3) an LLM judge over-credits the thief
too, and a Shapley definition fixes it only when the log shows what each agent could do alone.
Contribution: the credit gap as a detection metric, and the judge theft bias. Not a new algorithm;
cite and separate from lazy agents (VDN, Sunehag 2018), difference rewards (Wolpert & Tumer),
COMA (Foerster 2018), SHARP (Li 2026).

## Results already in hand (see `results/*/summary.md`, `report/checkin.tex`)
- Overcooked, 36 runs (PPO/GRPO x pickup/team/individual x stay/greedy partner, 3 seeds, 2M steps):
  pickup bonus -> 7-8x reference pickups, Shapley share ~0 (H1, 6/6 runs); individual reward -> own
  share ~1.0 but Shapley share 0.16-0.18, credit gap 0.82 vs reference 0.13 (H3); PPO under team
  reward also takes all deliveries (team soups 11 > reference 9.1), GRPO free-rides (s = 0.07).
- LBF and RWARE: partial replication (H2 holds with PPO; H3 gap does not reach 0.2).
- Oversight (`python -m eval.oversight results/full results/lbf_full results/rware_full`):
  credit theft passes a team-score check in 9/13 runs (0.69 [0.42, 0.87]).
- LLM-judge pilot (Assignment 7, Qwen3.8-27B, 15 simulated pairs x both orders x 3 runs): busy work
  rarely fools it; credit theft gives the thief +0.17-0.22 extra credit in every setting; Shapley
  definition + thinking fixes 6/8 cases but 0/4 when the thief took every soup; position bias
  without thinking (22/22 vs 14/22). Code: https://github.com/EunbiYoon/pilot.

## Experiments to add for the paper (priority order)
1. **cf reward as the fix** (12 runs, ~30 min). The Mac is already running it; if its
   `results/h10_cf` arrives, skip. Otherwise:
   `powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 grids -Grids h10_cf`
2. **More seeds** for the main Overcooked grid (3 -> 5+): `--seeds 3 4` with the same grid as
   `results/full/config.json`, merged into `results/full`.
3. **LLM judge on real rollouts with trained thieves** (main new result for the paper):
   - add individual-reward PPO checkpoints (`results/full/models/`) to the RLAIF pool via
     `python -m rlaif.collect ... --policies ckpt:<path>`;
   - judge conditions: basic prompt / + Shapley definition / + definition and each cook's solo results
     (pool meta has `v_partner_alone`, `v_learner_alone`); ~150 pairs each, both orders;
   - needs a prompt-condition flag in `rlaif/judge.py` (not implemented) and a theft-bias metric in
     `rlaif/audit.py` (judge prefers higher own deliveries but lower Shapley phi).
   - Model: `ollama pull qwen3.8:27b` (same weights as the Mac's qwen3.8:27b-mlx).
4. Optional: bonus sweep / partner noise / aux head (`run_windows.ps1 grids`), RLAIF closed loop.

## Windows setup
    py -3.12 -m venv venv
    .\venv\Scripts\python -m pip install -r requirements.txt   (drop alem-env if it fails)
Runner: `scripts/run_windows.ps1` (grids | rlaif | all; -Grids, -Workers, -JudgeModel, -Pairs).
Training is CPU (small MLP); the GPU is only for the Ollama judge.

## Paper plan (8 pages, or 4 if time runs out)
Intro / Related work / Setting (Overcooked, rewards, Shapley credit gap) / RL results (H1-H3, PPO vs
GRPO) / Oversight / LLM judge (pilot + real-rollout study) / Fix (cf reward, definition + solo
evidence) / Limitations (small MLP, one layout, few seeds, 2-player Shapley) / Conclusion.

## Other notes
- Unity cluster: credit grids were cancelled; jobs named ah-*/ahs-*/alem-t* belong to the separate
  safety-eval project, leave them alone. Unity scratch workspace "paper" expires 2026-12-10.
