from fphishe.evaluate import sensitivity_to_first_fp

SCCS = {
    "q": "a.1.1.1",
    "fam": "a.1.1.1",  # same family
    "rem": "a.1.1.2",  # same superfamily, different family
    "ign": "a.1.2.1",  # same fold, different superfamily: ignored
    "fp": "b.1.1.1",  # different fold
}


def test_levels():
    ranked = ["q", "fam", "ign", "fp", "rem"]
    assert sensitivity_to_first_fp("q", ranked, SCCS) == 0.5
    assert sensitivity_to_first_fp("q", ranked, SCCS, "family") == 1.0
    assert sensitivity_to_first_fp("q", ranked, SCCS, "remote") == 0.0
    assert sensitivity_to_first_fp("fp", ranked, SCCS) is None
