"""Summarize every results/hits_*.tsv: sensitivity to first FP, recall@k, timings, plot."""

import argparse
import glob
import os
import shutil
import subprocess
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


def defined(xs):
    """Drop queries the metric is undefined for (no true homologs at that level)."""
    return np.array([x for x in xs if x is not None])


sens, sens_family, sens_remote, recall = {}, {}, {}, {}
for path in sorted(glob.glob("results/hits_*.tsv")):
    method = os.path.basename(path)[len("hits_"):-len(".tsv")]
    hits = read_hits(path)
    ranked = {q: [t for t, _ in hits.get(q, [])] for q in queries}  # missing query -> no hits
    sens[method] = defined(sensitivity_to_first_fp(q, ranked[q], sccs) for q in queries)
    sens_family[method] = defined(
        sensitivity_to_first_fp(q, ranked[q], sccs, "family") for q in queries)
    sens_remote[method] = defined(
        sensitivity_to_first_fp(q, ranked[q], sccs, "remote") for q in queries)
    recall[method] = defined(recall_at_k(q, ranked[q], sccs, args.k) for q in queries)

timings = {}
if os.path.exists("results/timings.tsv"):
    for line in Path("results/timings.tsv").read_text().splitlines():
        m, sec, n, *device = line.split("\t")  # older files have no device column
        timings[m] = (float(sec), int(n), device[0] if device else "?")  # last run wins

rows = []
for m, s in sens.items():
    lo, hi = bootstrap_ci(s)
    pval = paired_permutation_test(sens["fphishe"], s) if "fphishe" in sens and m != "fphishe" else ""
    sec, n, device = timings.get(m, ("", "", ""))
    rows.append((m, len(s), s.mean(), lo, hi, pval, sens_family[m].mean(),
                 sens_remote[m].mean(), recall[m].mean(), sec, n, device))

header = ("method", "n", "sens_mean", "ci_lo", "ci_hi", "p_vs_fphishe", "sens_family",
          "sens_remote", f"recall@{args.k}", "seconds", "n_queries", "device")
with open("results/summary.tsv", "w") as f:
    f.write("\t".join(header) + "\n")
    f.writelines("\t".join(str(x) for x in row) + "\n" for row in rows)

print(f"{'method':<12}{'n':>5}{'sens':>8}{'95% CI':>16}{'p vs fphishe':>14}"
      f"{'family':>8}{'remote':>8}{f'recall@{args.k}':>12}")
for m, n, avg, lo, hi, pval, fam, rem, rec, *_ in rows:
    pv = f"{pval:.4f}" if pval != "" else "-"
    print(f"{m:<12}{n:>5}{avg:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>16}{pv:>14}"
          f"{fam:>8.3f}{rem:>8.3f}{rec:>12.3f}")
any_sens = next(iter(sens_family), None)
if any_sens:
    print(f"family: same-family homologs only ({len(sens_family[any_sens])} queries have one); "
          f"remote: same superfamily, different family ({len(sens_remote[any_sens])} queries)")

gpu = "none"
if shutil.which("nvidia-smi"):
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False).stdout.strip()
print(f"\ntimings (search only, data loading excluded; {os.cpu_count()} CPUs, GPU: {gpu})")
for m, (sec, n, device) in timings.items():
    print(f"{m:<14}{sec:>10.2f}s{n:>6} items{1000 * sec / n:>10.2f} ms/item   {device}")
if "fphishe" in timings and "embed" in timings:
    search_ms = 1000 * timings["fphishe"][0] / timings["fphishe"][1]
    embed_ms = 1000 * timings["embed"][0] / timings["embed"][1]
    print(f"fphishe end to end (query embedding + search): {search_ms + embed_ms:.2f} ms/query")

for m, s in sens.items():
    x = np.sort(s)[::-1]
    plt.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=m)
plt.xlim(0, 1)
plt.xlabel("sensitivity to first false positive")
plt.ylabel("fraction of queries with sensitivity >= x")
plt.legend()
plt.savefig("results/sensitivity.png", dpi=150, bbox_inches="tight")
