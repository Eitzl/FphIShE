"""GPU Smith-Waterman vs the Biopython reference, run in numba's CUDA simulator."""

import os
import random

import pytest

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")  # must be set before numba is imported
pytest.importorskip("numba")

from fphishe.align import sw_score
from fphishe.gpu import prepare, sw_scores

AA = "ACDEFGHIKLMNPQRSTVWY"


def test_matches_biopython():
    rng = random.Random(0)
    targets = [
        "A",
        "W",
        "X",
        "XXXXX",
        "AXCXW",
        "HEAGAWGHEE",
        "PAWHEAE",
        "MKTAYIAKQRQISFVKSHFSRQ",
        "MKTAYIAKQISFVKSHFSRQLEERLGLIEVQ",  # gaps relative to the previous one
        "MKTAYAAAAAAAAAAAAAIAKQRQISFVKSHFSRQ",  # long insertion
        "ACDEFGHIKLMNPQRSTVWYBZ*",
    ] + ["".join(rng.choice(AA) for _ in range(rng.randint(1, 120))) for _ in range(30)]
    queries = ["A", "X", "MKTAYIAKQRQISFVKSHFSRQ", "HEAGAWGHEE", targets[-1]]
    db = prepare(targets)
    for q in queries:
        got = sw_scores(q, db)
        want = [int(sw_score(q, t)) for t in targets]
        assert got.tolist() == want, q


def test_empty():
    db = prepare(["", "ACD"])
    assert sw_scores("ACD", db).tolist() == [0, int(sw_score("ACD", "ACD"))]
    assert sw_scores("", db).tolist() == [0, 0]
    assert len(sw_scores("ACD", prepare([]))) == 0
