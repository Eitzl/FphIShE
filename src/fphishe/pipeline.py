"""End-to-end two-stage search: prefilter then rerank."""

import numpy as np

from fphishe import index as ix
from fphishe.align import sw_score


def search(index, db_ids, db_seqs, Q, query_seqs, k=100, nprobe=10, score=sw_score):
    """Prefilter a batch of queries (rows of Q) to k candidates each, then rerank by alignment.

    Returns (prefilter, reranked): one [(db_id, score)] list per query, best first. Prefilter
    scores are negative L2 distances in the reduced embedding space.
    """
    D, rows = ix.search(index, Q, k, nprobe)
    prefilter = [
        [(db_ids[r], -float(np.sqrt(d))) for d, r in zip(dq, rq) if r != -1]
        for dq, rq in zip(D, rows)
    ]
    reranked = [
        sorted(((t, score(s, db_seqs[t])) for t, _ in p), key=lambda h: h[1], reverse=True)
        for s, p in zip(query_seqs, prefilter)
    ]
    return prefilter, reranked
