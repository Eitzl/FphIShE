"""CPU Smith-Waterman reference and parasail rerank."""

from Bio.Align import PairwiseAligner, substitution_matrices

GAP_OPEN = 11
GAP_EXTEND = 1

_aligner = PairwiseAligner(
    mode="local",
    substitution_matrix=substitution_matrices.load("BLOSUM62"),
    open_gap_score=-GAP_OPEN,
    extend_gap_score=-GAP_EXTEND,
)


def sw_score(a, b):
    """Reference local alignment score (Biopython)."""
    return _aligner.score(a, b)


def parasail_score(a, b):
    """Fast SIMD local alignment score. Needs the `align` extra."""
    import parasail

    return parasail.sw_striped_16(a, b, GAP_OPEN, GAP_EXTEND, parasail.blosum62).score
