"""Check GPU Smith-Waterman against the Biopython reference and time it against parasail.

On a GPU:        python scripts/check_gpu.py
In the simulator: NUMBA_ENABLE_CUDASIM=1 python scripts/check_gpu.py --small
"""

import argparse
import os
import random
import time

import numpy as np
from numba import cuda

from fphishe.align import sw_score
from fphishe.gpu import prepare, sw_scores

AA = "ACDEFGHIKLMNPQRSTVWYX"

p = argparse.ArgumentParser()
p.add_argument("--fasta", default="data/raw/astral40.fa")
p.add_argument("--queries", type=int, default=5)
p.add_argument("--check", type=int, default=300, help="targets compared against Biopython")
p.add_argument("--small", action="store_true", help="tiny random sequences, for the simulator")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
rng = random.Random(args.seed)
sim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"
print("device:", "CUDA simulator (CPU)" if sim else cuda.get_current_device().name)


def mutate(s):
    """Copy of s with random substitutions, insertions and deletions, so alignments have gaps."""
    out = []
    for c in s:
        r = rng.random()
        if r < 0.1:
            out.append(rng.choice(AA))
        elif r < 0.15:
            out.append(c + "".join(rng.choice(AA) for _ in range(rng.randint(1, 8))))
        elif r > 0.95:
            continue
        else:
            out.append(c)
    return "".join(out) or "A"


if args.small or not os.path.exists(args.fasta):
    print("using random sequences" + ("" if args.small else f" ({args.fasta} not found)"))
    max_len = 40 if args.small else 300
    n = 40 if args.small else 2000
    base = ["".join(rng.choice(AA) for _ in range(rng.randint(1, max_len))) for _ in range(n // 2)]
    db_seqs = base + [mutate(s) for s in base]
    args.queries = min(args.queries, 3) if args.small else args.queries
else:
    from fphishe.io import read_astral

    db_seqs = list(read_astral(args.fasta)[0].values())
    print(f"using {args.fasta}")
lens = [len(s) for s in db_seqs]
print(f"db: {len(db_seqs)} seqs, length min/median/max {min(lens)}/{int(np.median(lens))}/{max(lens)}")
queries = rng.sample(db_seqs, args.queries)

t0 = time.perf_counter()
sw_scores("ACDEFGHIK", prepare(["ACDEFGHIK"]))  # first call compiles the kernel (warm-up)
cuda.synchronize()
print(f"JIT compile + first call: {time.perf_counter() - t0:.3f}s")

# Correctness: every query against a random subset of the db.
check = rng.sample(db_seqs, min(args.check, len(db_seqs)))
check_db = prepare(check)
bad = 0
for qi, q in enumerate(queries):
    got = sw_scores(q, check_db)
    for ti, t in enumerate(check):
        want = int(sw_score(q, t))
        if got[ti] != want:
            bad += 1
            if bad <= 20:
                print(f"MISMATCH query {qi} (len {len(q)}) vs target {ti} (len {len(t)}): "
                      f"expected {want}, got {got[ti]}\n  q={q}\n  t={t}")
print(f"correctness: {bad} mismatches out of {len(queries) * len(check)} pairs")

# Speed: one query against the full db.
t0 = time.perf_counter()
db = prepare(db_seqs)
cuda.synchronize()
print(f"prepare (encode + upload): {time.perf_counter() - t0:.3f}s")

q = queries[0]
t0 = time.perf_counter()
scores = sw_scores(q, db)
cuda.synchronize()
gpu = time.perf_counter() - t0
cells = len(q) * sum(lens)
print(f"GPU: query len {len(q)} vs {len(db_seqs)} targets: {gpu:.3f}s ({cells / gpu / 1e9:.2f} GCUPS)")
print(f"  top scores: {sorted(scores.tolist(), reverse=True)[:5]}")

try:
    from fphishe.align import parasail_score

    t0 = time.perf_counter()
    for t in db_seqs:
        parasail_score(q, t)
    cpu = time.perf_counter() - t0
    print(f"parasail (1 CPU core): {cpu:.3f}s ({cells / cpu / 1e9:.2f} GCUPS), GPU speedup {cpu / gpu:.1f}x")
except ImportError:
    print("parasail not installed; skipping CPU timing (pip install parasail)")
