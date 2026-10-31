"""FphIShE: PCA + FAISS IVF prefilter, then parasail Smith-Waterman rerank of the top k."""

import argparse
import statistics
import time
from pathlib import Path

import faiss

from fphishe import pipeline
from fphishe.align import parasail_score
from fphishe.index import build_index
from fphishe.io import read_astral, read_embeddings, write_hits
from fphishe.reduce import fit_pca, transform

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=128)
p.add_argument("--k", type=int, default=100)
p.add_argument("--nlist", type=int, default=100)
p.add_argument("--nprobe", type=int, default=10)
p.add_argument("--metric", choices=["l2", "cos"], default="l2")
p.add_argument("--repeats", type=int, default=1, help="time the search this many times")
args = p.parse_args()

seqs, _ = read_astral("data/raw/astral40.fa")
ids, X = read_embeddings("data/processed/embeddings.h5")
queries = Path("data/processed/queries.txt").read_text().split()
row = {d: i for i, d in enumerate(ids)}

t0 = time.perf_counter()
Z = transform(fit_pca(X, args.dim), X)
if args.metric == "cos":
    faiss.normalize_L2(Z)  # in place; cosine of the PCA-centered vectors
index = build_index(Z, args.nlist, args.metric)
build = time.perf_counter() - t0

times = []
for _ in range(args.repeats):
    t0 = time.perf_counter()
    prefilter, reranked = pipeline.search(
        index, ids, seqs, Z[[row[q] for q in queries]], [seqs[q] for q in queries],
        args.k, args.nprobe, parasail_score,
    )
    times.append(time.perf_counter() - t0)
elapsed = statistics.median(times)

write_hits("results/hits_prefilter.tsv", dict(zip(queries, prefilter)))
write_hits("results/hits_fphishe.tsv", dict(zip(queries, reranked)))
# Columns: method, seconds, queries, device. The rerank runs on one core; FAISS may use more.
with open("results/timings.tsv", "a") as f:
    f.write(f"fphishe_build\t{build:.3f}\t{len(queries)}\tcpu\n")
    f.write(f"fphishe\t{elapsed:.3f}\t{len(queries)}\tcpu x1\n")
print(f"build {build:.1f}s, search {elapsed:.1f}s (median of {args.repeats}) for "
      f"{len(queries)} queries")
