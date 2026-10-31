import faiss
import numpy as np
import pytest

from fphishe import pipeline
from fphishe.index import build_index


@pytest.mark.parametrize("metric, self_score", [("l2", 0.0), ("cos", 1.0)])
def test_search_finds_self_first(metric, self_score):
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(50)]
    seqs = {d: "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 60)) for d in ids}
    X = rng.normal(size=(50, 8)).astype(np.float32)
    if metric == "cos":
        faiss.normalize_L2(X)
    index = build_index(X, nlist=2, metric=metric)
    prefilter, reranked = pipeline.search(
        index, ids, seqs, X[:3], [seqs[d] for d in ids[:3]], k=5, nprobe=2
    )
    for q, p, r in zip(ids, prefilter, reranked):
        assert p[0][0] == q and p[0][1] == pytest.approx(self_score, abs=1e-5)
        assert r[0][0] == q
        assert len(p) == len(r) == 5
