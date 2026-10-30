"""Benchmarks against SCOP labels: sensitivity to first false positive, recall@k.

A hit is a true positive if it shares the query's superfamily, a false positive if it is in a
different fold, and ignored otherwise (same fold, different superfamily: homology unclear).
Self hits are always ignored. Both metrics return None for queries with no true homologs.
"""


def label(q_sccs, t_sccs):
    """+1 true positive, -1 false positive, 0 ignored."""
    q, t = q_sccs.split("."), t_sccs.split(".")
    if q[:3] == t[:3]:
        return 1
    if q[:2] != t[:2]:
        return -1
    return 0


def n_true(query, sccs):
    return sum(label(sccs[query], s) == 1 for t, s in sccs.items() if t != query)


def sensitivity_to_first_fp(query, ranked_targets, sccs):
    """Fraction of the query's true homologs ranked above its first false positive."""
    total = n_true(query, sccs)
    if total == 0:
        return None
    tp = 0
    for t in ranked_targets:
        if t == query:
            continue
        lab = label(sccs[query], sccs[t])
        if lab == -1:
            break
        tp += lab == 1
    return tp / total


def recall_at_k(query, ranked_targets, sccs, k):
    """Fraction of the query's true homologs in the top k (self excluded)."""
    total = n_true(query, sccs)
    if total == 0:
        return None
    top = [t for t in ranked_targets if t != query][:k]
    return sum(label(sccs[query], sccs[t]) == 1 for t in top) / total
