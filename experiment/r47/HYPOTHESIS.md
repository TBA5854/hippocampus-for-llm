# r47 — does a single-token word-key dissolve the char-level ceiling?

r36/r44: the char-level language ceiling is the multi-token-key tax — a fact key `ab` is 2 tokens, the retrieval
point is the shared `:`, so pos0 is a 1/n_facts draw, and 8 facts asymptotes ~80% even fully trained. A BPE key is a
single token, which is exactly the property MQAR uses to reach 100%.

- **H47a — bpe_recall at 8 facts works COLD**, without the curriculum char-level needed. Predicted: exact well above
  the char-level cold 10.3%, ideally near MQAR's 100%. Falsified if it is at chance cold — that would mean the real
  shared vocabulary (word keys competing with prose embeddings), not the multi-token layout, was the problem.
- **H47b — controls at chance.** `--hc off` and `--shuffle-store` cannot answer; `--facts-in-context` is the ceiling.
- Chance is ~1/|val_pool| (hundreds of word tokens), so exact match is a clean single-token metric here.

Single-token keys are real words in the SAME vocabulary as the distractor prose, so unlike MQAR (synthetic
out-of-vocab symbols) this tests whether exact addressing survives a shared, correlated embedding space.
