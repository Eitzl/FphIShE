"""Benchmarks against SCOP labels: sensitivity to first false positive, recall@k.

A hit is a true positive if it shares the query's superfamily, a false positive if it is in a
different fold, and ignored otherwise (same fold, different superfamily: homology unclear).
Self hits are always ignored. Both metrics return None for queries with no true homologs.

`level` restricts which true positives count: "family" (same family, close homologs) or
"remote" (same superfamily, different family). None counts all of them.
"""


def label(q_sccs, t_sccs):
    """+1 true positive, -1 false positive, 0 ignored."""
    q, t = q_sccs.split("."), t_sccs.split(".")
    if q[:3] == t[:3]:
        return 1
    if q[:2] != t[:2]:
        return -1
    return 0


def counts(q_sccs, t_sccs, level):
    """Whether this pair is a true positive at the given level."""
    if label(q_sccs, t_sccs) != 1:
        return False
    if level is None:
        return True
    same_family = q_sccs.split(".")[:4] == t_sccs.split(".")[:4]
    return same_family == (level == "family")


_n_true = {}  # every method is scored against the same labels, so count once


def n_true(query, sccs, level=None):
    key = (id(sccs), query, level)
    if key not in _n_true:
        _n_true[key] = sum(counts(sccs[query], s, level) for t, s in sccs.items() if t != query)
    return _n_true[key]


def sensitivity_to_first_fp(query, ranked_targets, sccs, level=None):
    """Fraction of the query's true homologs ranked above its first false positive."""
    total = n_true(query, sccs, level)
    if total == 0:
        return None
    tp = 0
    for t in ranked_targets:
        if t == query:
            continue
        if label(sccs[query], sccs[t]) == -1:
            break
        tp += counts(sccs[query], sccs[t], level)
    return tp / total


def recall_at_k(query, ranked_targets, sccs, k):
    """Fraction of the query's true homologs in the top k (self excluded)."""
    total = n_true(query, sccs)
    if total == 0:
        return None
    top = [t for t in ranked_targets if t != query][:k]
    return sum(label(sccs[query], sccs[t]) == 1 for t in top) / total
