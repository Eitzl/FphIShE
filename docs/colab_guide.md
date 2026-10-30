# Running the FphIShE MVP on Google Colab

This guide takes you from an empty Colab session to a benchmark table and plot. The notebook
`notebooks/run_mvp.ipynb` has the same steps as cells, with the same section numbers.

This is the first end-to-end run of an ultra-rough MVP. The GPU code has never run on real
hardware, so plan on test-fix cycles in section 5. Fill in the checklist at the end as you go.

## What you're about to run

FphIShE is a two-stage protein homology search. ProtT5 turns each protein into a 1024-d vector.
PCA reduces those vectors, and a FAISS IVF index returns the k nearest neighbours of each query
(the prefilter). Smith-Waterman (parasail) then rescores those k candidates and reorders them.
The benchmark runs this on SCOPe 2.08 ASTRAL 40% (15177 domains). It compares the result against
full Smith-Waterman, which aligns every query against all 15177 domains (CPU parasail, and a numba
CUDA kernel), and against BLAST.

```
download_data.py   ASTRAL FASTA ──► data/raw/astral40.fa
                                         │
embed.py           ProtT5 (GPU) ──► data/processed/embeddings.h5   (slow step; back it up)
                                         │
pick_queries.py    500 random queries ──► data/processed/queries.txt
                                         │
          ┌──────────────────────────────┼─────────────────────────────┐
search_fphishe.py                 search_sw.py [--gpu]           search_blast.py
PCA → IVF top-k → SW rerank       all-vs-all SW                  blastp
hits_prefilter.tsv                hits_sw.tsv / hits_sw_gpu.tsv  hits_blast.tsv
hits_fphishe.tsv
          └──────────────────────────────┼─────────────────────────────┘
                                         │
report.py          results/summary.tsv, results/sensitivity.png, printed table + timings
```

Every script runs from the repo root and needs no arguments. All scripts except
`download_data.py` and `embed.py` use relative paths, so a different working directory breaks
them.

Rough time budget on a free T4: setup about 5 min, embedding about 10–30 min (estimate, never
measured), everything else a few minutes.

## 0. Colab setup

### 0.1 GPU runtime

Runtime → Change runtime type → **T4 GPU** → Save. Then check it:

```
!nvidia-smi
```

You should see a Tesla T4. If you get "command not found", the runtime has no GPU.

### 0.2 GitHub token in Colab Secrets

The repo (https://github.com/Eitzl/FphIShE) is private, so cloning needs a personal access token.

1. On GitHub, go to Settings → Developer settings → Personal access tokens → Fine-grained tokens →
   Generate new token. For repository access, pick "Only select repositories" → `Eitzl/FphIShE`.
   For permissions, set Repository → Contents → **Read-only**. Pick a short expiry.
2. In Colab, click the key icon in the left sidebar (Secrets) → Add new secret. Name it
   `GH_TOKEN`, paste the token as the value, and turn on "Notebook access".

Never paste the token into a cell, print it, or commit it. Shared notebooks and cell outputs can
leak it.

### 0.3 Clone

```python
from google.colab import userdata
TOKEN = userdata.get("GH_TOKEN")
!git clone -b feature/work-space https://$TOKEN@github.com/Eitzl/FphIShE.git /content/FphIShE
# Remove the token from .git/config so it isn't stored on disk
!git -C /content/FphIShE remote set-url origin https://github.com/Eitzl/FphIShE.git
%cd /content/FphIShE
!git log --oneline -3
```

IPython replaces `$TOKEN` with the Python variable, so the token never appears in the cell text.

To pull fixes later in the same session (you will need this in the GPU test-fix loop):

```python
TOKEN = userdata.get("GH_TOKEN")
!git -C /content/FphIShE pull https://$TOKEN@github.com/Eitzl/FphIShE.git feature/work-space
```

If you edited a file locally (for example `MAX_RESIDUES` in `embed.py`), run
`!git checkout -- scripts/embed.py` first, or the pull refuses to merge.

### 0.4 Install

```
%cd /content/FphIShE
!pip install -q -e ".[align,gpu,embed,dev]"
!apt-get -qq install -y ncbi-blast+
```

Colab already has torch, so pip keeps it. Then sanity-check the install:

```
!python -c "import fphishe, faiss, parasail, numba, torch, transformers, h5py; print('numba', numba.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
!blastp -version
!pytest -q
```

`pytest` runs the GPU test in numba's CUDA simulator on the CPU, so it passes even if real CUDA is
broken. All tests should pass.

Run every script as `!python scripts/...` from `/content/FphIShE`. Each one runs in a fresh
subprocess, which sees the editable install without restarting the kernel. A new session (after a
disconnect or "Disconnect and delete runtime") loses the clone and the install, so redo all of
section 0. A plain "Restart session" keeps them.

### 0.5 Google Drive (for persistence)

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/fphishe
```

The only expensive artifact is `embeddings.h5` (section 2). Everything else is regenerated in
minutes.

## 1. Data

```
!python scripts/download_data.py
```

This downloads the 4.3 MB ASTRAL FASTA to `data/raw/astral40.fa` and should print
`15177 sequences`. Sequence lengths have a median of 151 and a max of 1664. Rerunning it skips the
download.

## 2. Embeddings

### 2.1 Restore a previous run (new sessions only)

If an earlier session saved embeddings to Drive, copy them back first. `embed.py` then only does
whatever is missing.

```
!mkdir -p data/processed
!cp -v /content/drive/MyDrive/fphishe/embeddings.h5 data/processed/ 2>/dev/null || echo "no saved embeddings"
```

### 2.2 Embed

```
!python scripts/embed.py
```

- The first run downloads `Rostlab/prot_t5_xl_half_uniref50-enc` (about 2.4 GB).
- It writes one 1024-d float32 mean-pooled vector per domain to `data/processed/embeddings.h5`.
  Estimates, not measured: about 10–30 min on a T4, and a 65–100 MB file.
- It processes the longest sequences first, so an out-of-memory error shows up in the first few
  batches rather than at the end.
- It is resumable: it skips ids already in the h5. If the session dies, rerun it.
- It prints `N already embedded, M to go on cuda`. If it says `cpu`, the runtime has no GPU. Stop
  and fix that, because CPU embedding takes hours.
- If you hit CUDA out of memory, lower `MAX_RESIDUES` (default 4000):
  `!sed -i 's/^MAX_RESIDUES = 4000/MAX_RESIDUES = 2000/' scripts/embed.py`, then rerun.
- If you see a misleading "tiktoken" error from the tokenizer, protobuf is missing (this happens
  with transformers 5). The `embed` extra includes it. Fix it with `!pip install protobuf sentencepiece`.

Record the wall time and file size. These are the most useful numbers from this run.

### 2.3 Back up to Drive

Write locally and then copy, because writing the h5 straight to Drive is slow and flaky.

```
!ls -lh data/processed/embeddings.h5
!python -c "import h5py; print(len(h5py.File('data/processed/embeddings.h5','r')), 'embedded (expect 15177)')"
!cp -v data/processed/embeddings.h5 /content/drive/MyDrive/fphishe/
```

Every query must be in the h5, so embed the full FASTA (15177) before moving on. If the session
died mid-embedding, also copy the partial file to Drive. It still resumes.

## 3. Queries

```
!python scripts/pick_queries.py            # defaults: --n 500 --seed 0
```

This writes `data/processed/queries.txt` with 500 domains, each with at least one other member of
its superfamily.

Re-running `pick_queries.py` with different `--n` or `--seed` makes every existing hits file
stale. The report counts a query missing from a hits file as sensitivity 0, so after picking new
queries, rerun every search in section 4 and 5.

For a clean run, also delete old timings. `results/timings.tsv` is append-only, and the report
uses the last line for each method.

```
!rm -f results/timings.tsv
```

## 4. Searches (CPU)

The three searches are independent, so run them in any order.

### 4.1 FphIShE

```
!python scripts/search_fphishe.py          # defaults: --dim 128 --k 100 --nlist 100 --nprobe 10
```

This writes `results/hits_prefilter.tsv` (the IVF ranking by embedding distance) and
`results/hits_fphishe.tsv` (the same k candidates reordered by parasail score). It prints build
time (PCA + index) and search time.

For reference, a local run (14-core CPU, random stand-in embeddings, 50 queries) took 2.4 s to
build and 0.1 s to search. Search cost depends on k, not on embedding quality, so expect the
same order of magnitude here.

### 4.2 Full Smith-Waterman, CPU

```
!python scripts/search_sw.py               # defaults: --top 1000, all CPU cores
```

This aligns every query against all 15177 domains with parasail and writes `results/hits_sw.tsv`
(top 1000 per query). It uses multiprocessing, so always run it as a script and never paste it into
a cell.

Locally it took 1.9 s for 50 queries with 14 workers. Colab has 2 vCPUs, so expect about 7× slower:
roughly 2 min for 500 queries. That is an extrapolation, not a measurement.

### 4.3 BLAST

```
!python scripts/search_blast.py            # defaults: --evalue 10 --max-targets 1000
!head -3 results/hits_blast.tsv
```

This writes `results/hits_blast.tsv`. The middle column must be a bare domain id like `d1dlwa_`,
the same form as in `hits_sw.tsv`. If it looks like `gnl|BL_ORD_ID|123` or similar, the BLAST
row would score as sensitivity 0. This script has not been tested with real BLAST, so check it.

## 5. GPU check, then GPU Smith-Waterman

This is the code that has never run on real hardware. Do the checks in order and stop at the
first failure.

### 5.1 Does numba see the GPU?

```
!python -c "from numba import cuda; cuda.detect()"
```

Expect `Found 1 CUDA devices` with a Tesla T4 marked `[SUPPORTED]`. If it fails at driver or NVVM
init:

```
!pip install -q "numba-cuda[cu12]"
```

Then rerun the detect cell. Newer numba releases moved the CUDA target into the separate
`numba-cuda` package.

Do not set `NUMBA_ENABLE_CUDASIM` in the notebook. If you set it with `os.environ`, every later
`!python` subprocess inherits it and silently runs in the CPU simulator.

### 5.2 Kernel logic in the simulator

```
!pytest -q tests/test_gpu.py
```

This runs on the CPU and checks the kernel's logic against Biopython. It should pass whether or
not CUDA works.

### 5.3 Real-hardware correctness and speed

```
!python scripts/check_gpu.py --queries 5 --check 300
```

This must report `correctness: 0 mismatches out of 1500 pairs`. It also prints:

- `JIT compile + first call`: one-time numba compile cost.
- `GPU: ... (X GCUPS)`: one query against the full database. GCUPS means billions of DP cells per
  second.
- `parasail (1 CPU core): ... GPU speedup Nx`.

Record all three numbers. The kernel uses the naive design (see section 9), so a modest speedup or
even a slowdown would not surprise anyone. Correctness matters first.

### 5.4 Full GPU search

Run this only after 5.3 shows 0 mismatches.

```
!python scripts/search_sw.py --gpu
```

This writes `results/hits_sw_gpu.tsv`. Its sensitivity in the report should match `sw` almost
exactly, because both compute the same scores. Only tie ordering can differ.

### 5.5 When something fails: what to send back to Claude

Copy all of the following into the conversation:

1. The full traceback or the full `check_gpu.py` output, including any MISMATCH lines. Don't
   trim it.
2. The output of `!python -c "from numba import cuda; cuda.detect()"`.
3. Versions: `!python -c "import numba; print(numba.__version__)"; pip list 2>/dev/null | grep -i -E "numba|cuda"`.
4. `!nvidia-smi` (the header line gives the driver and CUDA version).

After a fix is pushed, pull it (section 0.3) and rerun from 5.2.

Known risks in the current kernel, to keep in mind when reading failures or timings:

- **Packaging:** numba vs numba-cuda. CUDA support is moving out of numba proper, and driver or
  NVVM init errors usually come from this.
- **Warp divergence:** targets aren't sorted by length, so the 32 threads in a warp loop over
  targets of very different lengths, and each warp waits for its longest target.
- **Low occupancy:** one thread per target is only about 15k threads, well below what a T4 can keep
  resident.
- **Scratch reallocation:** `H` and `E` (query_len × n_targets int32) are allocated on every
  `sw_scores` call, which means once per query.
- **Global-memory-bound access:** the DP rows, the query, and BLOSUM62 are all read from global
  memory in the inner loop.

## 6. Report and how to read it

```
!python scripts/report.py                  # --k must match the fphishe --k (default 100)
```

```python
from IPython.display import Image
Image("results/sensitivity.png")
```

The report reads every `results/hits_*.tsv`, so expect one row per method: `blast`, `fphishe`,
`prefilter`, `sw`, and `sw_gpu` if you ran it. It also writes `results/summary.tsv`. Delete stale
hits files (for example a `hits_sw_gpu.tsv` from a broken run) or they show up as rows.

### Columns

- **sens**: mean *sensitivity to first false positive*. For each query, walk down its ranked hits.
  Count true positives (same SCOP superfamily) until you reach the first false positive (a
  different fold). Divide by the number of true homologs of that query in the whole database.
  Same fold but different superfamily is ignored (the homology is unclear), and so is the
  self-hit. 1.0 means every homolog ranks above every clear non-homolog.
- **95% CI**: bootstrap CI of that mean over queries.
- **p vs fphishe**: two-sided paired permutation (sign-flip) test on the per-query differences
  between this method and fphishe. A small p means the difference isn't noise from 500 queries.
- **recall@k**: fraction of each query's true homologs in its top k (self excluded). The
  `prefilter` row is the key one. FphIShE only reorders the prefilter's k candidates, so a homolog
  the prefilter missed can never come back. fphishe's recall@k equals prefilter's (same set), and
  fphishe's sensitivity can't exceed it. If prefilter recall@k is low, the fix is in the embedding
  or index (k, dim, nprobe), not the rerank.

### Timings

The report prints `seconds`, `n queries`, and `ms/query` per method, for search only (data loading
excluded).

- `fphishe` is the prefilter plus the parasail rerank.
- `fphishe_build` is PCA + index build, and `blast_build` is `makeblastdb`. Both are one-time costs
  divided by the number of queries, so their ms/query isn't a meaningful per-query cost.
- `sw_gpu` excludes the GPU upload and JIT compile (a warm-up call runs before the timer).
- Embedding time is not in the table. Record it yourself from section 2.

### What "this worked" roughly looks like

- `sw` sensitivity is plausible. A local run with random stand-in embeddings gave full-SW
  sensitivity 0.25 on these queries, a reasonable sanity number for ASTRAL 40%. Your `sw` row
  should be in that neighbourhood, since it doesn't depend on embeddings.
- `sw_gpu` ≈ `sw`.
- `fphishe` lands well above `prefilter`'s sensitivity, and is competitive with `sw` and `blast`.
- `prefilter` recall@k is clearly far from zero. With random embeddings, fphishe was about 0, so a
  near-zero fphishe now means the embeddings didn't load or match. Check that
  `embeddings.h5` has 15177 entries and that it came from this FASTA.

No expected numbers exist yet for real embeddings. This run produces the first ones.

### Save results

```
!cp -rv results /content/drive/MyDrive/fphishe/results_$(date +%Y%m%d_%H%M)
```

## 7. Experiments worth trying

Each one is a single flag. Rerun the affected search, then the report.

| Knob | Command | What it tells you |
|---|---|---|
| prefilter size k | `search_fphishe.py --k 50` (and `report.py --k 50`) | Recall vs rerank cost. Larger k raises the ceiling on fphishe sensitivity, and the rerank cost grows linearly. |
| PCA dim | `search_fphishe.py --dim 64` / `256` / `512` | How much of ProtT5's 1024-d space the neighbour search needs. |
| IVF nprobe | `search_fphishe.py --nprobe 1` / `25` / `100` | Recall lost to approximate search. `--nprobe 100` with `--nlist 100` searches every list, which is exact. |
| queries n | `pick_queries.py --n 1000`, then rerun all searches | Tighter CIs. This makes all hits stale (section 3). |

A k sweep that keeps each summary:

```
!for k in 25 50 100 200 400; do python scripts/search_fphishe.py --k $k && python scripts/report.py --k $k && cp results/summary.tsv results/summary_k$k.tsv; done
```

`report.py --k` also changes the recall@k column for `sw` and `blast`, which keeps the columns
comparable across rows. Each run overwrites `hits_fphishe.tsv` and `sensitivity.png`.

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `git clone` asks for a password, or returns 403 / "Authentication failed" | Token missing, expired, or not granted this repo | Regenerate the fine-grained token with Contents: Read on `Eitzl/FphIShE`. If a fine-grained token can't see the repo (for example, you're a collaborator, not the owner), use a classic token with `repo` scope as a fallback. |
| `SecretNotFoundError` / `NotebookAccessError` from `userdata.get` | Secret not created, misnamed, or notebook access off | Key icon → check the name `GH_TOKEN` and the "Notebook access" toggle |
| `nvidia-smi: command not found`, or embed says `on cpu` | No GPU runtime | Runtime → Change runtime type → T4 GPU (this resets the session) |
| `FileNotFoundError: data/raw/astral40.fa` or `data/processed/...` | Not in the repo root | `%cd /content/FphIShE` |
| `ModuleNotFoundError: fphishe` | Install not run, or a new session | Redo section 0.4 |
| Tokenizer error mentioning `tiktoken` | protobuf missing (transformers 5) | `!pip install protobuf sentencepiece` |
| `CUDA out of memory` in `embed.py` | Batch too large for a T4 | Lower `MAX_RESIDUES` (section 2.2) |
| `OSError: Unable to open file` / truncated h5 | Session died mid-write | Restore the Drive copy, or delete the h5 and re-embed |
| `KeyError: 'd...'` in `search_fphishe.py` | A query isn't in `embeddings.h5` | Finish `embed.py` (expect 15177 entries) |
| `makeblastdb: not found` | BLAST not installed | `!apt-get -qq install -y ncbi-blast+` |
| `blast` sensitivity ≈ 0 | `sseqid` isn't the bare domain id | `!head results/hits_blast.tsv`, then report it |
| A method's sensitivity suddenly drops | Stale hits after a new `pick_queries.py` | Rerun every search |
| Timings look like an old run | `timings.tsv` is append-only | `!rm -f results/timings.tsv` and rerun the searches |
| `CudaSupportError`, NVVM, or libdevice errors | numba / numba-cuda packaging | `!pip install -q "numba-cuda[cu12]"`, then section 5.5 |
| `check_gpu.py` prints MISMATCH | Kernel bug on real hardware | Send the full output (section 5.5) |
| `check_gpu.py` says "CUDA simulator" | `NUMBA_ENABLE_CUDASIM=1` is set | Restart the session without setting it |
| `search_sw.py` is very slow | 2 vCPUs | Expected (about 2 min, estimate). Use `--gpu` or fewer queries |
| `search_sw.py` hangs or pickling errors when pasted into a cell | multiprocessing needs a script | Run it with `!python scripts/search_sw.py` |

## 9. Known rough edges and optimization candidates

These come from reading the code. Nothing here is fixed yet.

**Cost and pipeline**

- **Embedding is the dominant cost.** Everything downstream takes seconds to minutes. ProtT5 is
  loaded in fp32 and then cast to fp16. Batching is a simple residue budget (`MAX_RESIDUES`), so a
  larger budget on a bigger GPU, or a smaller/faster PLM, are the levers.
- **`read_embeddings` reads 15177 separate HDF5 datasets** one at a time. Storing a single
  (N, 1024) array would load faster (not measured).
- **PCA is fit on the full database, queries included.** That's fine for a benchmark, but it's
  transductive. A real deployment fits PCA on the database only.
- **IVF `nlist`/`nprobe` aren't tuned.** At 15k vectors an exact `IndexFlatL2` search is cheap, so
  IVF only matters at larger scale. `--nprobe 100` shows the recall cost of the approximation.
- **The self-hit takes one of the k slots.** Queries come from the database, so `--k 100` gives 99
  real candidates.
- **The fphishe list is only k long, while SW and BLAST keep up to 1000.** fphishe sensitivity is
  capped by prefilter recall@k (section 6), which is by design but worth knowing when comparing
  rows.

**Rerank and alignment**

- **`pipeline.search` reranks with Python-level parasail calls,** one per (query, candidate) pair,
  with no query-profile reuse. A parasail profile built once per query (`profile_create_16` +
  `sw_striped_profile_16`) would cut per-call overhead. The same applies to `search_sw.py`.
- **The fphishe rerank doesn't use the GPU.** All k candidates per query could go to the GPU kernel
  in one batch.

**GPU kernel** (`src/fphishe/gpu/__init__.py`)

- Possible numba vs numba-cuda packaging trouble on Colab.
- Warp divergence from unsorted target lengths. Sorting targets by length in `prepare` is cheap.
- Low occupancy: one thread per target is about 15k threads.
- `H`/`E` scratch is reallocated on every call (once per query). It could be allocated once at the
  max query length.
- Global-memory-bound: the DP row, query, and BLOSUM62 are all in global memory. Shared or constant
  memory, a query profile, or an intra-task (anti-diagonal / striped) design are the standard next
  steps.
- One kernel launch plus a synchronous copy back per query. Queries could be batched.

**Report and bookkeeping**

- **The report recomputes per-query homolog counts for each method:** `evaluate.n_true` is O(N)
  and runs twice per query per method. `pick_queries.py` already counts superfamilies once.
  Locally the report took about 2.5 s for 50 queries, so it scales to tens of seconds at 500
  (estimate).
- `timings.tsv` is append-only, and the last line per method wins.
- `report.py --k` must be matched to `search_fphishe.py --k` by hand.
- Scripts other than download and embed use cwd-relative paths.
- `search_blast.py` rebuilds the BLAST database on every run and uses `os.cpu_count()` threads
  (2 on Colab).

## Questions to note as you go

Fill this in during the run. It's the input for the next round of fixes.

| Step | Wall time | Problems / confusion |
|---|---|---|
| 0 Setup (clone + install) | | |
| 1 Data | | |
| 2 Embeddings (also: h5 size, MAX_RESIDUES used) | | |
| 3 Queries | | |
| 4.1 search_fphishe (build / search) | | |
| 4.2 search_sw CPU | | |
| 4.3 search_blast | | |
| 5.3 check_gpu (mismatches / JIT / GCUPS / speedup) | | |
| 5.4 search_sw --gpu | | |
| 6 report | | |

Also note:

- [ ] Where did you have to guess what to do next?
- [ ] Which error messages were misleading?
- [ ] Did any step take far longer than this guide estimated?
- [ ] Is prefilter recall@k high enough that the rerank has something to work with?
- [ ] Did `sw_gpu` match `sw`?
- [ ] Which result surprised you?
- [ ] What would you want to change before the next run?
