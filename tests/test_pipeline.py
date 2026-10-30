import numpy as np

from fphishe import pipeline
from fphishe.index import build_index


def test_search_finds_self_first():
    rng = np.random.default_rng(0)
    ids = [f"d{i}" for i in range(50)]
    seqs = {d: "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 60)) for d in ids}
    X = rng.normal(size=(50, 8)).astype(np.float32)
    index = build_index(X, nlist=2)
    prefilter, reranked = pipeline.search(
        index, ids, seqs, X[:3], [seqs[d] for d in ids[:3]], k=5, nprobe=2
    )
    for q, p, r in zip(ids, prefilter, reranked):
        assert p[0] == (q, 0.0) and r[0][0] == q
        assert len(p) == len(r) == 5
