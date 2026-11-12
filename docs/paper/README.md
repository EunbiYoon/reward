# AAAI-27 workshop paper: what is where, and what is left before Nov 20

Target: WMAC 2027 (https://multiagents.org), deadline 2026-11-20 23:59 UTC (confirm on the site).
Backup: Effective and Efficient Evaluation for AI Scientists, 2026-11-20 06:00 UTC.
All AAAI-27 workshops require an author to present in person in Montreal (Feb 22-23, 2027).

## Files
- `docs/paper/main.tex`: draft (abstract, intro, related work, setting, 4 result sections, discussion,
  15 references). Red `\PENDING{}` marks numbers that come from the running experiments.
- `docs/paper/figures/`: made by `scripts/paper_figures.py` (fig_rl, fig_judge, fig_bonus, fig_noise).
- `scripts/paper_experiments.sh`: every experiment the paper uses (started 2026-11-11 21:08).

## Experiments (running on the Mac, `tail -f logs/paper.log`)
| step | output | feeds |
|---|---|---|
| main grid seeds 3-4 (24 runs) | results/paper/full5 | Table 1, Fig 1 (5 seeds) |
| cf reward seeds 3-4 (8 runs) | results/paper/cf5 | Section "difference reward" |
| LLM judge, 200 pairs x 3 prompts | results/rlaif/theft/audit.md | Table 2, Fig 2 |
| pickup-bonus sweep (30 runs) | results/h7_bonus | optional figure / appendix |
| partner-noise sweep (48 runs) | results/h8_noise | optional figure / appendix |
| oversight on everything | results/paper/oversight.md | Section "supervisor" |

If it stops (sleep, crash), rerun the same command: finished runs and judged pairs are skipped.

    cd ~/Desktop/credit_hacking && caffeinate -i nohup bash scripts/paper_experiments.sh > logs/paper.log 2>&1 &

## Checklist to submit
1. When `logs/paper.log` says `all done`: fill every `\PENDING{}` from the files above
   (ask Claude: "실험 끝났어, 논문 채워줘").
2. Move the body into the official AAAI-27 author kit (aaai27.sty / aaai27.bst); check the page limit
   (WMAC: up to 8 pages full / 4 short) and whether the venue is double-blind (`\anontrue` hides the name).
3. Read the related-work citations for the 2025-2026 arXiv papers (Shihab, Zhang, Li, Powell) and add
   full author lists and arXiv ids.
4. Submit on OpenReview / the workshop's system before the deadline.
