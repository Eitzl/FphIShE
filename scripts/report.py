"""Summarize every results/hits_*.tsv: sensitivity to first FP, recall@k, timings, plot."""

import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fphishe.evaluate import recall_at_k, sensitivity_to_first_fp
from fphishe.io import read_astral, read_hits
from fphishe.stats import bootstrap_ci, paired_permutation_test

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=100, help="recall@k cutoff (the fphishe k)")
args = p.parse_args()

_, sccs = read_astral("data/raw/astral40.fa")
queries = Path("data/processed/queries.txt").read_text().split()

sens, recall = {}, {}
for path in sorted(glob.glob("results/hits_*.tsv")):
    method = os.path.basename(path)[len("hits_"):-len(".tsv")]
    hits = read_hits(path)
    ranked = {q: [t for t, _ in hits.get(q, [])] for q in queries}  # missing query -> no hits
    s = [sensitivity_to_first_fp(q, ranked[q], sccs) for q in queries]
    r = [recall_at_k(q, ranked[q], sccs, args.k) for q in queries]
    sens[method] = np.array([x for x in s if x is not None])
    recall[method] = np.array([x for x in r if x is not None])

timings = {}
if os.path.exists("results/timings.tsv"):
    for line in Path("results/timings.tsv").read_text().splitlines():
        m, sec, n = line.split("\t")
        timings[m] = (float(sec), int(n))  # last run of each method wins

rows = []
for m, s in sens.items():
    lo, hi = bootstrap_ci(s)
    pval = paired_permutation_test(sens["fphishe"], s) if "fphishe" in sens and m != "fphishe" else ""
    sec, n = timings.get(m, ("", ""))
    rows.append((m, len(s), s.mean(), lo, hi, pval, recall[m].mean(), sec, n))

header = ("method", "n", "sens_mean", "ci_lo", "ci_hi", "p_vs_fphishe", f"recall@{args.k}",
          "seconds", "n_queries")
with open("results/summary.tsv", "w") as f:
    f.write("\t".join(header) + "\n")
    f.writelines("\t".join(str(x) for x in row) + "\n" for row in rows)

print(f"{'method':<12}{'n':>5}{'sens':>8}{'95% CI':>16}{'p vs fphishe':>14}"
      f"{f'recall@{args.k}':>12}")
for m, n, mean, lo, hi, pval, rec, _, _ in rows:
    pv = f"{pval:.4f}" if pval != "" else "-"
    print(f"{m:<12}{n:>5}{mean:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>16}{pv:>14}{rec:>12.3f}")
print("\ntimings (search only, data loading excluded)")
for m, (sec, n) in timings.items():
    print(f"{m:<14}{sec:>10.2f}s{n:>6} queries{1000 * sec / n:>10.2f} ms/query")

for m, s in sens.items():
    x = np.sort(s)[::-1]
    plt.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=m)
plt.xlim(0, 1)
plt.xlabel("sensitivity to first false positive")
plt.ylabel("fraction of queries with sensitivity >= x")
plt.legend()
plt.savefig("results/sensitivity.png", dpi=150, bbox_inches="tight")
