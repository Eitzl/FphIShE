"""CPU Smith-Waterman reference and parasail rerank."""

import numpy as np
import parasail


def sw_scores(query, db_seqs, gap_open=11, gap_extend=1):
    """Local alignment scores (BLOSUM62) of `query` against each sequence in `db_seqs`.

    parasail charges a gap of length L as gap_open + (L - 1) * gap_extend.
    """
    profile = parasail.profile_create_sat(query, parasail.blosum62)
    return np.array(
        [parasail.sw_striped_profile_sat(profile, t, gap_open, gap_extend).score for t in db_seqs],
        dtype=np.int32,
    )
