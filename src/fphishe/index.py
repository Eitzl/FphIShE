"""Embedding prefilter: FAISS IVF index build and search."""

import faiss


def build_index(X, nlist=100, metric="l2"):
    """metric "l2" ranks by Euclidean distance; "cos" by cosine similarity (inner product of
    unit vectors), so with "cos" X and the queries must be L2-normalized first."""
    d = X.shape[1]
    if metric == "cos":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, nlist)
    index.train(X)
    index.add(X)
    return index


def search(index, Q, k=100, nprobe=10):
    """Returns (distances, row indices), each shape (n_queries, k). Missing hits are -1."""
    index.nprobe = nprobe
    return index.search(Q, k)
