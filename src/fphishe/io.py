"""Loading sequences (FASTA) and embeddings (HDF5)."""

import h5py
import numpy as np
from Bio import SeqIO


def read_fasta(path):
    return {r.id: str(r.seq) for r in SeqIO.parse(path, "fasta")}


def read_embeddings(path):
    """ProtT5 per-protein HDF5: one dataset per sequence id."""
    with h5py.File(path, "r") as f:
        ids = list(f.keys())
        X = np.stack([f[i][:] for i in ids]).astype(np.float32)
    return ids, X
