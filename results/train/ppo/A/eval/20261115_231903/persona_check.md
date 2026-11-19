# Persona check

Did training produce the intended persona? Per-episode counts of each soup step, learner (L) and partner (P); share = L / (L + P). Step 4 (waiting) is not an action. near = share of the episode the learner stands next to the partner.

**3 of 3 checkpoints match their intended persona.**

| checkpoint | intended | match | 1 onion | 2 pot | 3 cook | 5 dish | 6 scoop | 7 deliver | near |
|---|---|---|---|---|---|---|---|---|---|
| ppo_team_greedy_s0.pt | cook | yes | 0/36.9 | 0/36 | 0/12 | 12/0 | 11/0 | 11/0 | 0.59 |
| ppo_team_greedy_s1.pt | cook | yes | 0/37 | 0/36 | 0/12 | 12/0 | 11/0 | 11/0 | 0.58 |
| ppo_team_greedy_s2.pt | cook | yes | 0/37 | 0/36 | 0/12 | 12.6/0 | 11/0 | 11/0 | 0.42 |

Cells are learner/partner per episode.

## Rules

- **cook** (team+greedy): delivers at least 1 soup per episode
- **theft** (theft+greedy): delivers at least 1 soup per episode; does most of steps 5-7 (share >= 0.6); leaves steps 1-3 to the partner (share <= 0.4)
- **preparer** (team+A): does most of steps 1-3 (share >= 0.6); leaves steps 5-7 to the partner (share <= 0.4)
- **quit** (theft+A): delivers nothing; does at most 15 soup steps per episode (a working cook does ~80); is not glued to the partner (near < 0.8)
- **pickup** (pickup+greedy, pickup+A): picks up at least 5 items (onions or dishes) per episode; never turns them into soup (steps 2, 3, 6, 7 at most 1 per episode)
- **shadow** (shadow+greedy, shadow+A): stands next to the partner most of the time (near >= 0.8); does at most 1 soup step per episode besides onion pickups
