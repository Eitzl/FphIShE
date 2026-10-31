"""Where does the prefilter lose homologs that full Smith-Waterman finds?

Takes every true homolog that sw ranks above its first false positive (the hits that count
toward sw's sensitivity) and checks whether it is in the prefilter's top k. A miss is blamed on
the first stage that drops it:
  ivf        in the exact top k of the PCA space, but IVF searched the wrong clusters
  pca        in the exact top k of the full 1024-d space, but not after PCA
  embedding  not in the top k even in the full space; also prints how deep those hits sit
Run search_sw.py first. Use the same flags as search_fphishe.py.
"""

import argparse
from pathlib import Path

import faiss
import numpy as np

from fphishe.evaluate import label
from fphishe.index import build_index
from fphishe.io import read_astral, read_embeddings, read_hits
from fphishe.reduce import fit_pca, transform

p = argparse.ArgumentParser()
p.add_argument("--dim", type=int, default=128)
p.add_argument("--k", type=int, default=100)
p.add_argument("--nlist", type=int, default=100)
p.add_argument("--nprobe", type=int, default=10)
p.add_argument("--metric", choices=["l2", "cos"], default="l2")
args = p.parse_args()

_, sccs = read_astral("data/raw/astral40.fa")
ids, X = read_embeddings("data/processed/embeddings.h5")
queries = Path("data/processed/queries.txt").read_text().split()
sw = read_hits("results/hits_sw.tsv")
row = {d: i for i, d in enumerate(ids)}
qrows = [row[q] for q in queries]

full = X - X.mean(0)  # PCA with every component kept: same distances as this
Z = transform(fit_pca(X, args.dim), X)
if args.metric == "cos":
    faiss.normalize_L2(full)
    faiss.normalize_L2(Z)
index = build_index(Z, args.nlist, args.metric)
index.nprobe = args.nprobe
_, ivf_rows = index.search(Z[qrows], args.k)


def exact_ranks(V):
    """Rank of every database row for every query, 0 = closest."""
    Q = V[qrows]
    D = -Q @ V.T if args.metric == "cos" else (Q**2).sum(1)[:, None] - 2 * Q @ V.T + (V**2).sum(1)
    return np.argsort(np.argsort(D, axis=1), axis=1)


rank_pca, rank_full = exact_ranks(Z), exact_ranks(full)

found = kept = 0
lost = {"ivf": 0, "pca": 0, "embedding": 0}
deep = []  # full-space ranks of the hits lost to the embedding
for i, q in enumerate(queries):
    in_ivf = set(ivf_rows[i])
    for t, _ in sw.get(q, []):
        if t == q:
            continue
        lab = label(sccs[q], sccs[t])
        if lab == -1:
            break
        if lab == 0:
            continue
        found += 1
        r = row[t]
        if r in in_ivf:
            kept += 1
        elif rank_pca[i, r] < args.k:
            lost["ivf"] += 1
        elif rank_full[i, r] < args.k:
            lost["pca"] += 1
        else:
            lost["embedding"] += 1
            deep.append(rank_full[i, r])

print(f"{found} true homologs ranked above sw's first false positive, over {len(queries)} queries")
print(f"  kept in prefilter top {args.k}: {kept} ({kept / found:.1%})")
for stage, n in lost.items():
    print(f"  lost to {stage:<10} {n:>6} ({n / found:.1%})")
if deep:
    q50, q90 = np.percentile(deep, [50, 90])
    print(f"  embedding-lost hits sit at full-space rank: median {q50:.0f}, 90th pct {q90:.0f}")
