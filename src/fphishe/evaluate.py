"""Benchmarks: sensitivity to first false positive, recall@k, timing."""


def sensitivity_to_first_fp(ranked_ids, true_ids):
    """Fraction of true homologs ranked above the first false positive."""
    tp = 0
    for i in ranked_ids:
        if i not in true_ids:
            break
        tp += 1
    return tp / len(true_ids)


def recall_at_k(ranked_ids, true_ids, k):
    return len(set(ranked_ids[:k]) & set(true_ids)) / len(true_ids)
