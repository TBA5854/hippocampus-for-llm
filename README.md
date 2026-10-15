# Memory as a block

Code, experiments and manuscripts for an in-stack episodic memory for transformers. One
persistent store is read and written at every layer and addressed by the raw, position-free
layer-0 token embedding, so retrieval is an exact token match. The weights are stateless and the
store is the state: a frozen model becomes a different agent with each store it loads.

## What is here

| path | what |
|---|---|
| `mech/` | the model, memory block, store, tasks, training and evaluation probes (PyTorch) |
| `tests_mech/` | unit tests for the memory block, tasks, tokenizer and data code |
| `experiment/` | one folder per experiment, `r1` to `r113`, each with `HYPOTHESIS.md`, `run.sh`, `RESULTS.md`, `results/` (JSON) and `logs/`. `experiment/README.md` indexes them and `experiment/STATE.md` records the setup |
| `paper/` | the long-form paper (LaTeX) |
| `journal/` | the condensed journal manuscript, its Supplementary Information, figure scripts and source data. See `journal/README.md` |

Experiment logs were copied from two GPU machines; the user name and address of the second one
are replaced with `<user>` and `<box2-host>`.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --group dev          # torch, tokenizers, pytest
uv run pytest                # runs tests_mech/
```

## Demo (no GPU, no data)

`demo/store_demo.py` exercises the store itself with a hand-built embedding table and no trained
model. It writes 64 facts, reads them back by exact token match, grows the store to about
100,000 slots, runs the controls, and saves, reloads and refuses a foreign store file.

```sh
uv run python demo/store_demo.py
```

Run time: about 30 seconds wall clock on a 10-core CPU (12 seconds inside the script, the rest is
importing torch). Expected output, with timings that vary:

```
1-2. 64 facts written once, queried by their own key tokens
     exact-match recall: 100.0%   (slots used: 64)
3.   store grown to 100,064 slots of other tokens; top-1 read (1 slot used per query)
     recall: 100.0%  [2.39s for 64 queries]
4.   controls
     scrambled store (another store read with these queries): 0.0%
     empty store: 0.0%
5.   saved, reloaded into a fresh store: recall 100.0%
     foreign model's file refused: store was written by model model-A, not model-B; its keys an...
```

This shows the store mechanics only. The trained-model results (a language model learning to use the
store) need GPU runs; they are in `experiment/` and the manuscript. Training the task from scratch
on a CPU is not practical: 300 steps took about 7 minutes and had not yet learned it.

## Data

The language experiments use PG-19 (Rae et al., ICLR 2020). `mech/data.py` reads
`<root>/<split>/*.txt` with PG-19's own split names; the runs here used `data/pg19`. The BPE
tokenizer (8,192 tokens) is trained on that text with `mech/bpe.py`. Neither the data nor the
trained checkpoints are in this repository.

## Reproducing results and figures

- Every experiment's exact command is in its `run.sh`; its numbers are in `results/*.json` and
  `RESULTS.md`. Re-running a round needs the PG-19 text, the tokenizer, and, for the evaluation
  probes, the checkpoint that round names.
- The journal figures and tables are rebuilt from the committed result files, with no GPU:

  ```sh
  cd journal
  python3 figures/make_figures.py     # needs matplotlib
  python3 figures/make_table1.py
  pdflatex main && bibtex main && pdflatex main && pdflatex main
  ```
