"""Embed data/raw/astral40.fa with ProtT5 into data/processed/embeddings.h5.

One float32 (1024,) dataset per domain id: the mean of the per-residue embeddings.
Resumable: ids already in the h5 file are skipped.

--ids/--out embed a subset into another file, e.g. the queries, to time what a new query costs.
The embedding time (model loading excluded) is appended to results/timings.tsv as "embed".
"""

import argparse
import re
import time
from pathlib import Path

import h5py
import torch
from transformers import T5EncoderModel, T5Tokenizer

from fphishe.io import read_astral

ROOT = Path(__file__).resolve().parents[1]
FASTA = ROOT / "data/raw/astral40.fa"
MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
MAX_RESIDUES = 4000  # per batch, counting padding; lower it if a T4 runs out of memory
PRINT_EVERY = 20  # batches

p = argparse.ArgumentParser()
p.add_argument("--ids", help="file of domain ids to embed (default: all)")
p.add_argument("--out", default=str(ROOT / "data/processed/embeddings.h5"))
args = p.parse_args()
OUT = Path(args.out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
tokenizer = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False)
model = T5EncoderModel.from_pretrained(MODEL).to(device=device, dtype=dtype).eval()

seqs, _ = read_astral(FASTA)
if args.ids:
    seqs = {i: seqs[i] for i in Path(args.ids).read_text().split()}
OUT.parent.mkdir(parents=True, exist_ok=True)
with h5py.File(OUT, "a") as f:
    todo = sorted((i for i in seqs if i not in f), key=lambda i: len(seqs[i]), reverse=True)
    print(f"{len(seqs) - len(todo)} already embedded, {len(todo)} to go on {device}")

    # Longest first, so an out-of-memory error shows up right away rather than hours in.
    batches, batch = [], []
    for i in todo:
        if batch and (len(batch) + 1) * len(seqs[batch[0]]) > MAX_RESIDUES:
            batches.append(batch)
            batch = []
        batch.append(i)
    if batch:
        batches.append(batch)

    t0 = time.perf_counter()
    for n, batch in enumerate(batches, 1):
        texts = [" ".join(re.sub(r"[UZOB]", "X", seqs[i])) for i in batch]
        tok = tokenizer(texts, padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**tok).last_hidden_state.float().cpu().numpy()
        # Each residue is one token, followed by </s> and then padding.
        for row, i in enumerate(batch):
            f.create_dataset(i, data=hidden[row, : len(seqs[i])].mean(0))
        f.flush()
        if n % PRINT_EVERY == 0 or n == len(batches):
            print(f"batch {n}/{len(batches)}", flush=True)
    elapsed = time.perf_counter() - t0

if todo:
    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results/timings.tsv", "a") as f:
        f.write(f"embed\t{elapsed:.3f}\t{len(todo)}\t{device.type}\n")
    print(f"embedded {len(todo)} in {elapsed:.1f}s ({1000 * elapsed / len(todo):.2f} ms/seq)")
