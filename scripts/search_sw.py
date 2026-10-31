"""Full Smith-Waterman baseline: every query against every database sequence."""

import argparse
import os
import statistics
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fphishe.align import parasail_score
from fphishe.io import read_astral, write_hits


def init(targets):
    global TARGETS
    TARGETS = targets


def scores(query):
    return np.array([parasail_score(query, t) for t in TARGETS])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top", type=int, default=1000, help="hits kept per query")
    p.add_argument("--workers", type=int, default=None, help="default: all CPU cores")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--repeats", type=int, default=1, help="time the search this many times")
    args = p.parse_args()

    seqs, _ = read_astral("data/raw/astral40.fa")
    queries = Path("data/processed/queries.txt").read_text().split()
    ids = list(seqs)
    targets = [seqs[d] for d in ids]
    query_seqs = [seqs[q] for q in queries]

    if args.gpu:
        from fphishe import gpu

        db = gpu.prepare(targets)
        gpu.sw_scores(query_seqs[0], db)  # warm-up: JIT compile stays out of the timing

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        if args.gpu:
            all_scores = [gpu.sw_scores(q, db) for q in query_seqs]
        else:
            with Pool(args.workers, init, (targets,)) as pool:
                all_scores = pool.map(scores, query_seqs, chunksize=1)
        times.append(time.perf_counter() - t0)
    elapsed = statistics.median(times)

    hits = {}
    for q, s in zip(queries, all_scores):
        top = np.argsort(-np.asarray(s))[: args.top]
        hits[q] = [(ids[i], float(s[i])) for i in top]

    method = "sw_gpu" if args.gpu else "sw"
    device = "gpu" if args.gpu else f"cpu x{args.workers or os.cpu_count()}"
    write_hits(f"results/hits_{method}.tsv", hits)
    with open("results/timings.tsv", "a") as f:
        f.write(f"{method}\t{elapsed:.3f}\t{len(queries)}\t{device}\n")
    print(f"{method}: {elapsed:.1f}s (median of {args.repeats}) for {len(queries)} x "
          f"{len(targets)} alignments on {device}")
