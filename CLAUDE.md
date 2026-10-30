# CLAUDE.md — project context for agents

FphIShE (Fast protein-homolog Indexed Search Engine) is Evan Fitzhugh's fall 2026 UCSC
bioinformatics course project. It is graded; final presentations start **Dec 1, 2026**.

## Coding principles

- **YAGNI (You ain't gonna need it).** Agents and humans tend to over-engineer. Think about
  whether something is actually required before building it.
- **Simple first.** The simplest solution will always be the best one. Don't create
  multi-layered helper function wrappers; write flat, direct logic.
- **Delete aggressively.** Less code = fewer bugs.

## Research question

Can an embedding prefilter (FAISS search over ProtT5 protein embeddings) plus a Smith-Waterman
rerank find remote protein homologs as accurately as full Smith-Waterman, in a fraction of the
time? Does it find homologs that BLAST misses?

## Pipeline

1. **Prefilter** — ProtT5 per-protein embeddings (1024-d) → PCA (`reduce.py`) → FAISS IVF
   index built with k-means (`index.py`) → top-k candidates per query.
2. **Rerank** — Smith-Waterman alignment of the query against only those k candidates
   (`align.py`). CPU via parasail is the **core**; a custom GPU kernel (`gpu/`) is a **stretch goal**.
3. `pipeline.py` joins the stages; `evaluate.py` and `stats.py` produce results.

## Data (none of it is in git)

| Dataset | Use | Source |
|---|---|---|
| SCOPe 2.08 ASTRAL, <40% identity (~15k domains, family/superfamily/fold labels) | Ground truth benchmark; embed with ProtT5 on Colab | SCOPe website |
| Swiss-Prot + UniProt precomputed ProtT5 per-protein embeddings (HDF5) | Scale demo | UniProt FTP |
| BLAST+ | Baseline | NCBI |

Raw files go in `data/raw/`, derived files in `data/processed/`, outputs in `results/`.
All are gitignored. Download and recreate them with scripts in `scripts/`.

## Evaluation

- **Methods compared:** BLAST, full Smith-Waterman (exact reference), embedding-only ANN, combined pipeline. HMMER is optional.
- **Metrics:** sensitivity up to the first false positive (SCOPe superfamily/fold as truth), recall@k, time per query.
- **Statistics (course techniques):** PCA, k-means/IVF, dynamic programming, Monte Carlo
  shuffles for score significance, paired permutation tests and bootstrap CIs on sensitivity,
  regression of run time vs. database size and query length, and a parameter sweep
  (nlist/nprobe, k, PCA dims) to get a speed-vs-accuracy Pareto curve.

## Scope and schedule

- **Core deliverable:** PCA + IVF prefilter + CPU Smith-Waterman, with all statistics. The grade must not depend on the GPU work.
- **GPU gate (~Nov 20):** add the GPU kernel (inter-sequence parallelism, CUDA or Numba) only
  if its scores match parasail exactly. Otherwise, present it as partial work.
- The course covers dynamic programming on Nov 17, after alignment work begins.
- Keep Thanksgiving weekend light. Figures and slides are due by Nov 30.

## Repo conventions

- `src/` layout; install with `pip install -e ".[dev,align]"`; run `ruff check .` and `pytest`. CI runs both on pushes to `main`/`develop` and on PRs.
- Branches: `main` = stable; `develop` = integration; work on `feat/<name>` branches merged into `develop` by PR.
- Never commit data, embeddings, indexes, or results (`.h5`, `.npy`, `.faiss` are ignored). GitHub rejects files over 100 MB.
- The GPU code must have a CPU reference and a test that compares scores exactly.

## Notes for agents

- This is graded coursework. Evan wants to understand the "why" and to write the core
  algorithms himself. Scaffolding, tooling, tests, and explanations are fine. Ask before
  writing substantial algorithm or analysis code.
- Cloud agent workspaces have **no GPU** and can't reach SCOPe/UniProt (network allowlist).
  Write download scripts and GPU code for Evan to run on Colab; don't claim to have run them.
- Never force-push or rewrite history without asking.
