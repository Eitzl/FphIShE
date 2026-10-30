"""Embedding prefilter: FAISS IVF index build and search."""

import faiss


def build_index(X, nlist=100):
    quantizer = faiss.IndexFlatL2(X.shape[1])
    index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist)
    index.train(X)
    index.add(X)
    return index


def search(index, Q, k=100, nprobe=10):
    """Returns (distances, row indices), each shape (n_queries, k). Missing hits are -1."""
    index.nprobe = nprobe
    return index.search(Q, k)
