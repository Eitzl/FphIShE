from fphishe.io import load_scope

FASTA = """\
>d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Ciliate (Paramecium caudatum) [TaxId: 5885]}
slfeqlggqaavqavtaqfyaniqadatvatffngidmpnqtnktaaflcaalggpnawt
grnlkevhanmgvsnaqfttvighlrsaltgagvaaalveqtvavaetvrgdvvtv
>d2gkma_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Mycobacterium tuberculosis [TaxId: 1773]}
gllsrlrkrepisiydkiggheaievvvedfyvrvladdqlsaffsgtnmsrlkgkqvef
>d1ux8a_ b.2.3.4 (A:) Some other domain {Bacillus subtilis [TaxId: 1423]}
mkvlaag
"""


def test_load_scope(tmp_path):
    path = tmp_path / "scope.fa"
    path.write_text(FASTA)
    recs = load_scope(path)

    assert [r["id"] for r in recs] == ["d1dlwa_", "d2gkma_", "d1ux8a_"]
    assert recs[0]["seq"].startswith("SLFEQLGG")
    assert recs[0]["seq"].endswith("VRGDVVTV")  # wrapped lines are joined
    assert recs[2]["seq"] == "MKVLAAG"
    assert recs[2]["family"] == "b.2.3.4"
    assert recs[2]["superfamily"] == "b.2.3"
    assert recs[2]["fold"] == "b.2"
