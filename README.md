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
