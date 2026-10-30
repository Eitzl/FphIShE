"""End-to-end two-stage search: prefilter then rerank."""

from fphishe import index as ix
from fphishe.align import sw_score


def search(index, db_ids, db_seqs, query_emb, query_seq, k=100, nprobe=10, score=sw_score):
    """Prefilter one query to k candidates, then rerank by alignment score.

    Returns [(db_id, score)] sorted best first.
    """
    _, rows = ix.search(index, query_emb[None, :], k, nprobe)
    hits = [(db_ids[r], score(query_seq, db_seqs[db_ids[r]])) for r in rows[0] if r != -1]
    return sorted(hits, key=lambda h: h[1], reverse=True)
