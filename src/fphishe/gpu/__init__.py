"""GPU Smith-Waterman: one CUDA thread per database sequence (numba).

Scores match fphishe.align.sw_score: BLOSUM62, affine gaps where a gap of length L
costs GAP_OPEN + (L-1) * GAP_EXTEND. Set NUMBA_ENABLE_CUDASIM=1 to run on the CPU.
"""

import numpy as np
from Bio.Align import substitution_matrices
from numba import cuda

from fphishe.align import GAP_EXTEND, GAP_OPEN

_blosum = substitution_matrices.load("BLOSUM62")
ALPHABET = _blosum.alphabet  # 'ARNDCQEGHILKMFPSTWYVBZX*'
BLOSUM = np.array(_blosum, dtype=np.int32)  # (24, 24), integer-valued

# Byte -> residue index. Anything outside the alphabet becomes X.
_CODE = np.full(256, ALPHABET.index("X"), dtype=np.int8)
for _i, _c in enumerate(ALPHABET):
    _CODE[ord(_c)] = _i

NEG = -(1 << 28)  # "minus infinity" that can still have penalties subtracted without overflow
THREADS = 128


def encode(seq):
    """Residue string -> int8 array of alphabet indices."""
    return _CODE[np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)]


@cuda.jit
def _sw_kernel(query, residues, offsets, lengths, blosum, H, E, out):
    # Thread t aligns the whole query against target t. Rows = target residues,
    # columns = query residues. H[:, t] and E[:, t] hold the previous row; F is scalar.
    # Scratch is laid out (query_len, n_targets) so a warp's 32 threads touch adjacent memory.
    t = cuda.grid(1)
    if t >= lengths.shape[0]:
        return
    m = query.shape[0]
    for j in range(m):
        H[j, t] = 0
        E[j, t] = NEG
    best = 0
    start = offsets[t]
    for i in range(lengths[t]):
        a = residues[start + i]
        diag = 0  # H[i-1][j-1]; column 0 is the zero boundary
        left = 0  # H[i][j-1]
        f = NEG  # best score ending in a gap along the query (horizontal)
        for j in range(m):
            up = H[j, t]  # H[i-1][j]
            e = max(E[j, t] - GAP_EXTEND, up - GAP_OPEN)  # gap along the target (vertical)
            f = max(f - GAP_EXTEND, left - GAP_OPEN)
            h = max(0, diag + blosum[a, query[j]], e, f)
            H[j, t] = h
            E[j, t] = e
            diag = up
            left = h
            best = max(best, h)
    out[t] = best


def prepare(targets):
    """Encode database sequences once and upload them. Returns an opaque db tuple."""
    lengths = np.array([len(s) for s in targets], dtype=np.int32)
    offsets = np.zeros(len(targets), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)[:-1]
    residues = encode("".join(targets)) if lengths.sum() else np.zeros(1, dtype=np.int8)
    return (
        cuda.to_device(residues),
        cuda.to_device(offsets),
        cuda.to_device(lengths),
        cuda.to_device(BLOSUM),
    )


def sw_scores(query, db):
    """Local alignment score of `query` against every db target, in target order (int32)."""
    residues, offsets, lengths, blosum = db
    n = lengths.shape[0]
    if n == 0 or len(query) == 0:
        return np.zeros(n, dtype=np.int32)
    q = cuda.to_device(encode(query))
    H = cuda.device_array((len(query), n), dtype=np.int32)
    E = cuda.device_array((len(query), n), dtype=np.int32)
    out = cuda.device_array(n, dtype=np.int32)
    blocks = (n + THREADS - 1) // THREADS
    _sw_kernel[blocks, THREADS](q, residues, offsets, lengths, blosum, H, E, out)
    return out.copy_to_host()
