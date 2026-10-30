"""Loading sequences (ASTRAL FASTA), embeddings (HDF5), and search hits (TSV)."""

import h5py
import numpy as np
from Bio import SeqIO


def read_astral(path):
    """Returns (seqs, sccs), both keyed by domain id.

    ASTRAL headers look like '>d1dlwa_ a.1.1.1 (A:) ...'; the second field is the SCOP class.
    """
    seqs, sccs = {}, {}
    for r in SeqIO.parse(path, "fasta"):
        seqs[r.id] = str(r.seq).upper()
        sccs[r.id] = r.description.split()[1]
    return seqs, sccs


def read_embeddings(path):
    """ProtT5 per-protein HDF5: one dataset per sequence id."""
    with h5py.File(path, "r") as f:
        ids = list(f.keys())
        X = np.stack([f[i][:] for i in ids]).astype(np.float32)
    return ids, X


def write_hits(path, hits):
    """hits: {query: [(target, score), ...]}. Written as TSV rows: query, target, score."""
    with open(path, "w") as f:
        for q, targets in hits.items():
            f.writelines(f"{q}\t{t}\t{s}\n" for t, s in targets)


def read_hits(path):
    """Inverse of write_hits. Each query's targets come back sorted best score first."""
    hits = {}
    with open(path) as f:
        for line in f:
            q, t, s = line.split("\t")
            hits.setdefault(q, []).append((t, float(s)))
    for targets in hits.values():
        targets.sort(key=lambda h: h[1], reverse=True)
    return hits
