"""Concatenate sharded result CSVs: python -m rlhack.experiments.merge out.csv shard1.csv shard2.csv ..."""
import csv
import sys


def main():
    out, shards = sys.argv[1], sys.argv[2:]
    rows = [r for p in shards for r in csv.DictReader(open(p))]
    rows.sort(key=lambda r: (int(r["seed"]), int(r["budget"]), r["env"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
