"""BLAST baseline: NCBI BLAST+ blastp of the queries against the full database."""

import argparse
import os
import statistics
import subprocess
import time
from pathlib import Path

from fphishe.io import read_astral, write_hits

p = argparse.ArgumentParser()
p.add_argument("--evalue", type=float, default=10)
p.add_argument("--max-targets", type=int, default=1000)
p.add_argument("--threads", type=int, default=os.cpu_count())
p.add_argument("--repeats", type=int, default=1, help="time the search this many times")
args = p.parse_args()

seqs, _ = read_astral("data/raw/astral40.fa")
queries = Path("data/processed/queries.txt").read_text().split()

# Bare '>id' headers so BLAST reports our ids verbatim.
with open("data/processed/blast_db.fa", "w") as f:
    f.writelines(f">{d}\n{s}\n" for d, s in seqs.items())
with open("data/processed/blast_queries.fa", "w") as f:
    f.writelines(f">{q}\n{seqs[q]}\n" for q in queries)

t0 = time.perf_counter()
db = "data/processed/blast_db"
subprocess.run(
    ["makeblastdb", "-in", "data/processed/blast_db.fa", "-dbtype", "prot", "-out", db],
    check=True, stdout=subprocess.DEVNULL,
)
build = time.perf_counter() - t0

times = []
for _ in range(args.repeats):
    t0 = time.perf_counter()
    out = subprocess.run(
        ["blastp", "-query", "data/processed/blast_queries.fa", "-db", db,
         "-outfmt", "6 qseqid sseqid bitscore", "-evalue", str(args.evalue),
         "-max_target_seqs", str(args.max_targets), "-num_threads", str(args.threads)],
        check=True, capture_output=True, text=True,
    ).stdout
    times.append(time.perf_counter() - t0)
elapsed = statistics.median(times)

# One row per HSP; keep the best bitscore per (query, target).
best = {}
for line in out.splitlines():
    q, t, s = line.split("\t")
    best[q, t] = max(best.get((q, t), 0.0), float(s))
hits = {}
for (q, t), s in best.items():
    hits.setdefault(q, []).append((t, s))

write_hits("results/hits_blast.tsv", hits)
with open("results/timings.tsv", "a") as f:
    f.write(f"blast_build\t{build:.3f}\t{len(queries)}\tcpu x1\n")
    f.write(f"blast\t{elapsed:.3f}\t{len(queries)}\tcpu x{args.threads}\n")
print(f"blast: {elapsed:.1f}s (median of {args.repeats}) for {len(queries)} queries, "
      f"{len(hits)} with hits")
