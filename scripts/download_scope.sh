#!/usr/bin/env bash
# Download the SCOPe 2.08 ASTRAL <40% identity domain sequences into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."
wget -nc -P data/raw \
    https://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa
