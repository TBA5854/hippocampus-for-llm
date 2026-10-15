# Paper: Memory as a Block

LaTeX write-up of the in-stack shared-KV episodic store (v0.4 objective + v0.5 CSA2).
One file per section under `sections/`, wired together by `main.tex`.

## Build

```
cd paper
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Layout

- `main.tex` — document + `\input` of each section
- `sections/abstract.tex`
- `sections/introduction.tex`
- `sections/method.tex`        — the memory block (read early, write late, layer-0 key, value skip, store)
- `sections/language.tex`      — cross-segment recall on PG-19 (multi-token-key tax, BPE, curriculum)
- `sections/longcontext.tex`   — store > context window; sparse retrieval (top-1 beats dense)
- `sections/shared_kv.tex`     — CSA2 axes: candidate pool, cross-layer reuse, FP4
- `sections/placement.tex`     — read/write placement ablations; the fact-count cliff
- `sections/related.tex`
- `sections/discussion.tex`    — limitations (measured vs literal, curriculum, exact-key constraint)
- `sections/conclusion.tex`
- `references.bib`

## Source of the numbers

Every claim traces to a round under `../experiment/` (see `../experiment/README.md`
rounds table and `../experiment/OBJECTIVE.md`). Key rounds: real-language recall
r47--r51; long-context + sparse r51/r61/r62; CSA2 pool + reuse r64; FP4 r65;
placement/cliff r56/r59/r60/r63.
