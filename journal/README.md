# Journal manuscript — target: Nature Machine Intelligence (Article)

`../paper/` is the long-form version (TMLR-ready). This folder is the condensed journal
version; venue reasoning is in [VENUES.md](VENUES.md).

| file | what |
|---|---|
| `main.tex` | manuscript: abstract ≤150 words, main text ≤3,500, Methods, 6 display items |
| `supplementary.tex` | Supplementary Information — wraps `../paper/sections/*`, so edit those, not a copy |
| `references.bib` | numbered references (`naturemag` style) |
| `figures/make_figures.py` | rebuilds Figs 2–4 from `../experiment/` results, with 95% Wilson error bars, and writes `source_data/Fig*.csv` |
| `figures/fig1_architecture.tex` | Fig. 1 as its own vector PDF (Helvetica, 6–8 pt) |
| `figures/make_table1.py` | Table 1 counts, intervals and `source_data/Table1.csv` |
| `source_data/` | one CSV per figure and for Table 1 (NMI encourages source data) |
| `cover_letter.md` | cover letter draft; placeholders in brackets |
| `NMI_GUIDELINES.md` | the journal's rules, with sources and a compliance table |

Current budget: abstract **146/150** words, main text **~2,270/3,500** (placeholders
excluded), so ~1,200 words remain for the scaling and baseline results; the abstract has room
for at most 4 more words, so a scaling sentence needs another to be cut.

## Build

```sh
python3 figures/make_figures.py                 # needs system matplotlib (not the uv venv)
python3 figures/make_table1.py                  # run from journal/figures, or with it on the path
(cd figures && pdflatex fig1_architecture)
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplementary && pdflatex supplementary
```

## Figure provenance

Figures follow the NMI figure rules (≤180 mm, 8 pt bold panel labels, 5–7 pt text, no
gridlines, black text). Every plotted value is read from a result JSON or a run log under `experiment/`, with one
exception marked `SOURCE: README` in the script: the d512 dense-read point at 128k tokens
(r102). Its log was overwritten because box1 and box2 wrote same-named files into
`r102/logs/`, and the box2 copy landed last. Fig. 2b plots recall *relative to each model's
own 16-bit recall*, because the keys+values series (r50_bpe64, 64 facts) and the keys/values
ablation (r50_bpe16_s2, 16 facts) come from different models.

## Pending — run when the box is back

1. **Recreate the box2 ssh helper.** It lived in the session scratchpad, which the reboot
   wiped.
2. **Scale to ~100M parameters.** Retrofit the memory block onto a pretrained Pythia-160M or
   SmolLM2-135M rather than training from scratch:
   - Both use rotary position embeddings, so the input embedding is already position-free.
     That gives the layer-0 exact key for free.
   - The read gate starts at zero, so the pretrained model is untouched at step 0.
   - Fine-tuning fits one 16 GB card and takes a few hours per run.
   - From scratch is the wrong path:
     - A 100M model fits in memory (bf16, with gradient accumulation if needed).
     - Our measured cost is 164M tokens in 84 min for the 7.8M model, so a same-budget run
       is roughly 3–6 h at good utilisation, and slower in practice: the 12-layer per-layer
       read was too slow to reach a first eval in 31 min (r106).
     - A properly trained LM needs ~2B tokens, 12× more again.
3. **Baselines** at matched size on the recall and passkey tasks: Memformer, RMT, and
   Memorizing Transformers.
4. **Benchmarks:** BABILong and RULER.
5. **Fill the placeholders:**
   - Fig. 5.
   - The "Scaling and comparison with memory-augmented baselines" subsection.
   - One abstract sentence, at most 13 words.
   - Every `\pending{}` marker.
6. **Front matter, every red `[PENDING]` in the PDF:**
   - confirm the shared affiliation (taken from the corresponding author's ORCID record) applies to all three authors;
   - confirm or edit the human-oversight sentence in "Use of AI tools";
   - Zenodo DOIs for data and code, licence, reviewer link;
   - acknowledgements, funding statement, CRediT statement, competing interests;
   - cover letter brackets (preprint, prior editor contact, suggested reviewers).
7. **Code README** per the software checklist (system requirements, install, a small demo with
   expected output and run time, reproduction steps); open the checklist form in Adobe Reader.
8. **Submit:** full submission to Nature Machine Intelligence; it does not accept presubmission
   enquiries. Work through the compliance table in [NMI_GUIDELINES.md](NMI_GUIDELINES.md) first.
