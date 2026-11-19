"""HSP-style learner pool (Yu et al., ICLR 2023, "Learning Zero-Shot Cooperation with Humans, Assuming
Humans Are Biased"), adapted to find credit hackers without choosing the rewards by hand.

    1. weights.py   sample N event-based hidden rewards (HSP Table 12, onion-only layouts)  -> rewards.txt
    2. train.run    train one PPO learner per reward next to greedy (scripts/cluster/hsp_node.sh)
    3. select.py    keep the K most diverse learners by expected event counts (HSP's greedy selection)
    4. pool         cluster the kept learners and flag the hackers (python -m pool <run> --keep <run>/selected.txt)

Departure from HSP: HSP trains each biased policy in self-play with a task-reward partner. Here every
learner trains next to the scripted greedy cook, who can make soup alone, so the Shapley share s stays
measurable (with two partners who cannot cook alone, s = 0.5 by formula).
"""
