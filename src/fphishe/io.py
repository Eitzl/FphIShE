"""Loading sequences (FASTA) and embeddings (HDF5)."""

from Bio import SeqIO


def load_scope(path):
    """Load a SCOPe ASTRAL FASTA into a list of dicts.

    Each dict has keys id, seq (uppercase), family, superfamily, fold. A header like
    `>d1dlwa_ a.1.1.1 (A:) ...` gives family a.1.1.1, superfamily a.1.1, fold a.1.
    """
    records = []
    for rec in SeqIO.parse(path, "fasta"):
        sccs = rec.description.split()[1]
        parts = sccs.split(".")
        records.append({
            "id": rec.id,
            "seq": str(rec.seq).upper(),
            "family": sccs,
            "superfamily": ".".join(parts[:3]),
            "fold": ".".join(parts[:2]),
        })
    return records
