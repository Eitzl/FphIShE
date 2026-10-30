"""Pick a random subset of ASTRAL domains that have at least one true homolog."""

import argparse
import random
from collections import Counter

from fphishe.io import read_astral

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

_, sccs = read_astral("data/raw/astral40.fa")
# evaluate.n_true(q) > 0 <=> some other domain shares q's superfamily; counting is O(N).
superfamily = {d: ".".join(s.split(".")[:3]) for d, s in sccs.items()}
counts = Counter(superfamily.values())
candidates = sorted(d for d in sccs if counts[superfamily[d]] > 1)
queries = random.Random(args.seed).sample(candidates, min(args.n, len(candidates)))

with open("data/processed/queries.txt", "w") as f:
    f.writelines(q + "\n" for q in queries)
print(f"{len(queries)} queries from {len(candidates)} candidates ({len(sccs)} domains)")
