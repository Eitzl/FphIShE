import numpy as np
import pytest

parasail = pytest.importorskip("parasail")
pytest.importorskip("numba")

align = pytest.importorskip("fphishe.align")
sw = pytest.importorskip("fphishe.gpu.sw")

AA = "ACDEFGHIKLMNPQRSTVWY"
ODD = "XBZUOJ*"  # ambiguity codes, rare residues, and stop


def random_seq(rng, n):
    seq = rng.choice(list(AA), size=n)
    odd = rng.random(n) < 0.05
    seq[odd] = rng.choice(list(ODD), size=odd.sum())
    return "".join(seq)


def test_matrix_matches_parasail():
    assert np.array_equal(sw.MATRIX, parasail.blosum62.matrix)


def test_scores_match_parasail():
    rng = np.random.default_rng(0)
    query = random_seq(rng, 80)
    db = [random_seq(rng, n) for n in rng.integers(1, 150, size=150)]
    # Homologs: mutated copies of the query so the gap and match paths get exercised.
    for _ in range(10):
        s = list(query)
        for k in sorted(rng.choice(len(s), size=6, replace=False), reverse=True):
            s[k : k + 1] = rng.choice(["", "G", "GG", "GGG", s[k], "W"])
        db.append("".join(s))
    db += [query, query.lower(), "A", "W", "X", "B" * 5, "U", query[:1]]

    assert np.array_equal(sw.sw_scores(query, db), align.sw_scores(query, db))


def test_edge_cases():
    for query, db in [
        ("A", ["A", "W", "AAAA", "X"]),
        ("WWWWWWWWWWWWWWWWWWWW", ["WWWWWWWWWW" + "P" * L + "WWWWWWWWWW" for L in range(1, 6)]),
        ("MKVLAX", ["MKVLAX", "BZBZ", "mkvlax"]),
    ]:
        assert np.array_equal(sw.sw_scores(query, db), align.sw_scores(query, db))
