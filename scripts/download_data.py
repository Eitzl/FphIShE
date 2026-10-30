"""Download the SCOPe 2.08 ASTRAL 40% genetic-domain sequence set to data/raw/astral40.fa."""

import urllib.request
from pathlib import Path

from fphishe.io import read_astral

URL = (
    "https://scop.berkeley.edu/downloads/scopeseq-2.08/"
    "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
)
OUT = Path(__file__).resolve().parents[1] / "data/raw/astral40.fa"

if not OUT.exists():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, OUT.with_suffix(".part"))
    OUT.with_suffix(".part").rename(OUT)

seqs, _ = read_astral(OUT)
print(f"{OUT}: {len(seqs)} sequences")
