# r47 — a single-token word-key works COLD: 49.8% vs char-level's 10.3%, and the char tax is gone

BPE token-level recall, 8 facts, **cold** (no curriculum), keys/values single real word-tokens in the PG-19
vocabulary (8192, 3799 single-token words), distractors BPE prose. Answer is one token, so char == exact.

| run | exact | answer CE | gate |
|---|---|---|---|
| **`r47_bpe_on` — memory** | **49.8%** | 4.834 | 0.396 |
| `r47_bpe_off` — no memory (floor) | 0.0% | 7.610 | — |
| `r47_bpe_shuf` — scrambled (control) | 0.0% | 7.599 | 0.045 |
| `r47_bpe_ceiling` — facts in context | 96.1% | 0.262 | 0.234 |
| chance (~1/1899) | 0.05% | ~7.55 | |

## Verdicts

- **H47a — single-token keys work cold, and remove the char tax. Confirmed.** 49.8% exact against char-level's
  **10.3% cold** at the same 8 facts — a **5x jump** — and the whole r36 structure is gone: char == exact (49.8/49.8),
  no pos0/pos1 split, no 1/n_facts draw. There is no delimiter to key on; the slot keyed on the single-token word
  `k_j` returns `v_j` directly. Put another way, **BPE cold (49.8%) ~ char-level WITH a curriculum at 6k (46.4%)**:
  the single-token key buys cold what the curriculum had to buy for char-level.
- **H47b — controls at chance. Confirmed, cleanly.** Floor and scrambled both at **0.0%** with CE at log-vocab
  (7.6), and the scrambled gate closes to 0.045. The ceiling (facts in context) is 96.1%, so the task is learnable
  and the instrument sound.

## Diagnosis — the two bottlenecks are now cleanly separated

r36/r37 split the language failure into two axes: (1) the multi-token-key / delimiter ambiguity, and (2) the
fact-count bootstrapping deadlock (r25). This round isolates them: **BPE removes axis 1 entirely** (cold 49.8%,
char==exact), leaving only axis 2 — the gap from 49.8% to the 96.1% ceiling is the bootstrapping cost, exactly what
a curriculum fixed for char-level. So the prediction is clean and testable: **BPE + curriculum should reach the
ceiling**, and unlike char-level it starts from 50% rather than 10%, and has no residual per-character tax to cap it.
That is r48.

MQAR caveat retired: these keys are real words competing with the prose embedding space, not synthetic out-of-vocab
symbols, and exact single-token addressing still works (49.8% cold, 96.1% ceiling). Correlated real-word embeddings
do not break it.

## What this does NOT show

- One seed, cold, 6k steps, 8 facts, one vocab size (8192). The ceiling push (more steps / curriculum) and scaling
  past the char 13-fact cap are r48.
- BPE prose perplexity is not evaluated here (the answer-weighted term dominates loss); this is a recall result, not
  an LM one.
