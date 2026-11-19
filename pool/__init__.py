"""Persona map: find reward hackers in finished runs without naming the personas first.

    python -m pool                                  # every evaluated run folder under results/
    python -m pool results/full results/ppo_B2 --k 5

Pipeline (one module per step, run in this order by __main__.py):
    load.py      one row per trained learner, from each run's latest results.csv (+ reference.csv, config.json)
    screen.py    hacker flags: credit_gap, unearned, team_loss, no_shapley
    features.py  six behaviour measures per learner, standardised
    cluster.py   k-means for k = 2..max_k, k picked by silhouette
    naming.py    a suggested persona name per cluster (rules on the cluster mean)
    report.py    hackers.csv and hackers.md (cluster profile, hacker candidates, all learners)
    plot.py      hackers.png (learners on the first two principal axes; x = hacker candidate)

See pool/README.md for what each number means.
"""
