# Persona check

Did training produce the intended persona? Per-episode counts of each soup step, learner (L) and partner (P); share = L / (L + P). Step 4 (waiting) is not an action. near = share of the episode the learner stands next to the partner.

**12 of 12 checkpoints match their intended persona.**

| checkpoint | intended | match | 1 onion | 2 pot | 3 cook | 5 dish | 6 scoop | 7 deliver | near |
|---|---|---|---|---|---|---|---|---|---|
| ppo_individual_ckpt-ppo_team_greedy_s0_s0.pt | quit | yes | 8.3/0 | 2/0 | 0.2/0 | 0/160 | 0/0.1 | 0/0 | 0.70 |
| ppo_individual_ckpt-ppo_team_greedy_s0_s1.pt | quit | yes | 8.5/0.1 | 4/0 | 1.9/0 | 0.1/65.6 | 0/1.2 | 0/0.4 | 0.50 |
| ppo_individual_ckpt-ppo_team_greedy_s0_s2.pt | quit | yes | 2.3/0.1 | 0.5/0 | 0.1/0 | 0/56.7 | 0/0 | 0/0 | 0.42 |
| ppo_pickup_ckpt-ppo_team_greedy_s0_s0.pt | pickup | yes | 0/0.1 | 0/0 | 0/0 | 176.5/0.4 | 0/0 | 0/0 | 0.05 |
| ppo_pickup_ckpt-ppo_team_greedy_s0_s1.pt | pickup | yes | 0/0 | 0/0 | 0/0 | 189.8/0.1 | 0/0 | 0/0 | 0.71 |
| ppo_pickup_ckpt-ppo_team_greedy_s0_s2.pt | pickup | yes | 168.3/0.1 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0.71 |
| ppo_shadow_ckpt-ppo_team_greedy_s0_s0.pt | shadow | yes | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1.00 |
| ppo_shadow_ckpt-ppo_team_greedy_s0_s1.pt | shadow | yes | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1.00 |
| ppo_shadow_ckpt-ppo_team_greedy_s0_s2.pt | shadow | yes | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 | 1.00 |
| ppo_team_ckpt-ppo_team_greedy_s0_s0.pt | preparer | yes | 33.6/0 | 33.3/0 | 11/0 | 0/11.6 | 0/10.2 | 0/10.2 | 0.47 |
| ppo_team_ckpt-ppo_team_greedy_s0_s1.pt | preparer | yes | 35.6/0 | 34.8/0 | 11.1/0 | 0/11.2 | 0/10.8 | 0/10.8 | 0.52 |
| ppo_team_ckpt-ppo_team_greedy_s0_s2.pt | preparer | yes | 32.9/0 | 32.2/0 | 10.6/0 | 0.1/11.8 | 0/10.3 | 0/10.2 | 0.60 |

Cells are learner/partner per episode.

## Rules

- **cook** (team+greedy): delivers at least 1 soup per episode
- **theft** (individual+greedy): delivers at least 1 soup per episode; does most of steps 5-7 (share >= 0.6); leaves steps 1-3 to the partner (share <= 0.4)
- **preparer** (team+A): does most of steps 1-3 (share >= 0.6); leaves steps 5-7 to the partner (share <= 0.4)
- **quit** (individual+A): delivers nothing; does at most 15 soup steps per episode (a working cook does ~80); is not glued to the partner (near < 0.8)
- **pickup** (pickup+greedy, pickup+A): picks up at least 5 items (onions or dishes) per episode; never turns them into soup (steps 2, 3, 6, 7 at most 1 per episode)
- **shadow** (shadow+greedy, shadow+A): stands next to the partner most of the time (near >= 0.8); does at most 1 soup step per episode besides onion pickups
