"""Numba CUDA Smith-Waterman: one thread scores the query against one database sequence."""

import numpy as np
from Bio.Align import substitution_matrices
from numba import cuda

THREADS_PER_BLOCK = 128
NEG = -(2**30)

# Same alphabet, order, and values as parasail.blosum62. Letters outside the alphabet
# (U, O, J, ...) map to '*', as parasail does.
_B62 = substitution_matrices.load("BLOSUM62")
MATRIX = np.array(_B62, dtype=np.int8)
LUT = np.full(256, _B62.alphabet.index("*"), dtype=np.int8)
for _k, _a in enumerate(_B62.alphabet):
    LUT[ord(_a)] = _k
    LUT[ord(_a.lower())] = _k


def encode(seq):
    """Residue string -> int8 indices into MATRIX."""
    return LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


@cuda.jit
def sw_kernel(query, db, offsets, lengths, matrix, gap_open, gap_extend, H, E, out):
    """Gotoh affine-gap local alignment with one rolling row over the query.

    H[i, t] holds H(i, j-1) and E[i, t] the best score ending in a gap in the query
    (vertical move) for thread t. A gap of length L costs gap_open + (L - 1) * gap_extend.
    """
    t = cuda.grid(1)
    if t >= lengths.shape[0]:
        return
    n = query.shape[0]
    for i in range(n):
        H[i, t] = 0
        E[i, t] = NEG

    best = 0
    off = offsets[t]
    for j in range(lengths[t]):
        r = db[off + j]
        diag = 0  # H(i-1, j-1)
        h_up = 0  # H(i-1, j)
        f = NEG  # best score ending in a gap in the target (horizontal move)
        for i in range(n):
            h_left = H[i, t]  # H(i, j-1)
            e = max(E[i, t] - gap_extend, h_left - gap_open)
            f = max(f - gap_extend, h_up - gap_open)
            h = max(0, diag + matrix[query[i], r], e, f)
            E[i, t] = e
            H[i, t] = h
            diag = h_left
            h_up = h
            best = max(best, h)
    out[t] = best


def sw_scores(query, db_seqs, gap_open=11, gap_extend=1):
    """Local alignment scores (BLOSUM62) of `query` against each sequence in `db_seqs`.

    Matches fphishe.align.sw_scores (parasail) exactly. Targets are sorted by length so
    threads in a warp do similar amounts of work; scores come back in input order.
    """
    order = np.argsort([len(s) for s in db_seqs], kind="stable")
    seqs = [encode(db_seqs[k]) for k in order]
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)
    offsets = np.zeros(len(seqs), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]
    db = np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.int8)
    q = encode(query)

    n_t = len(seqs)
    # Rows = query positions, columns = threads, so neighbouring threads hit neighbouring memory.
    H = cuda.device_array((len(q), n_t), dtype=np.int32)
    E = cuda.device_array((len(q), n_t), dtype=np.int32)
    out = cuda.device_array(n_t, dtype=np.int32)
    blocks = (n_t + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    sw_kernel[blocks, THREADS_PER_BLOCK](
        cuda.to_device(q), cuda.to_device(db), cuda.to_device(offsets), cuda.to_device(lengths),
        cuda.to_device(MATRIX), gap_open, gap_extend, H, E, out,
    )

    scores = np.empty(n_t, dtype=np.int32)
    scores[order] = out.copy_to_host()
    return scores
