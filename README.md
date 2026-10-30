# FphIShE
Fast protein-homolog Indexed Search Engine

A two-stage protein homology search: an approximate nearest-neighbor prefilter over ProtT5
embeddings (PCA + FAISS IVF), then a Smith-Waterman rerank of the top candidates.
Benchmarked on SCOPe/ASTRAL against BLAST and full Smith-Waterman.

## Layout

```
src/fphishe/     package code
  io.py          FASTA / HDF5 loading
  reduce.py      PCA
  index.py       FAISS IVF prefilter
  align.py       CPU Smith-Waterman / parasail rerank
  pipeline.py    two-stage search
  evaluate.py    sensitivity, recall@k, timing
  stats.py       permutation tests, bootstrap CIs, regression
  gpu/           GPU Smith-Waterman (stretch goal)
tests/           pytest tests
scripts/         data download and experiment scripts
notebooks/       exploration and Colab notebooks
data/            raw/ and processed/ (gitignored)
results/         figures and tables (gitignored)
docs/            notes and write-up
```

## Setup

```bash
pip install -e ".[dev,align]"
pytest
```

## Branches

- `main`: stable, runnable code
- `develop`: integration branch; feature branches (`feat/...`) merge here by PR
