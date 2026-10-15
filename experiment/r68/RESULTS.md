# r68 — multi-hop through the shared store: RESULTS

**Verdict: multi-hop chaining did NOT work. All four arms sit at chance (≈0%);
the models never fit the 2-hop task at all (train loss 13.7→12.8, val ppl ~1e6).**
A clean negative — written up, not re-tuned.

## Numbers (2-hop recall: query head A, target tail C; best AND final)

| arm | best | final |
|-----|------|-------|
| hop_retrieved (`--hop-key retrieved`) | 0.3% | 0.3% |
| hop_token (control, no chaining) | 0.2% | 0.0% |
| hop_broken (control, chain cut) | 0.0% | 0.0% |
| hop_residual (`--hop-key residual`) | 0.2% | 0.0% |

All warm from r66_mlcold_f16 (99.7% single-hop @16), `bpe_hop` n_hops 2,
`--recency-read`, ml placement, 6k/4k steps, seed 0. hop_retrieved trajectory:
0.0 → 0.0 → 0.0 → 0.3% across 6k steps — nothing moved.

## What this rules in and out
- The controls behaving at 0% is *expected* (token can only reach B not C; broken
  has no path), so they say nothing on their own. The informative failure is
  **hop_retrieved ≈ 0**: feeding read layer L's retrieval as layer L+1's key did
  not chain A→B→C.
- The models did not fit the task at all (loss barely moved), so the base signal is
  either absent or unlearnable in this setup — not merely "chaining is imperfect".

## Leading hypothesis for the failure
The retrieved value is not a clean key. With `value_skip_norm` a stored value is
`scale·(norm(last_block) + norm(block0_skip))` — it carries the value token's
embedding but with an equal-magnitude last-block component. Fed back as the next
hop's query and matched against keys = token embeddings, that extra component
roughly halves the cosine to the intended key, so the exact-match that all recall
depends on no longer fires. The very property that lets top-1 sparse read work
(exact-token keys) is what a raw retrieved value cannot satisfy.

## What r69 must check before concluding (diagnosis, not re-tuning)
1. **Is the hop-task format learnable at all?** `n_hops 1` (query A → target B) =
   plain recall in the pad-separated triple format. If this fails too, the format
   or warm-start is the problem, not chaining.
2. **Does a curriculum enable 2-hop?** Warm 2-hop from a 1-hop hop-task checkpoint
   (compose after the base is established), mirroring how curriculum unlocked
   fact-count recall (r49/r66).
3. **Does more training alone help?** A long (12k) cold-warm 2-hop run.

If 1-hop works but 2-hop stays at chance under both curriculum and long training,
the conclusion is that a raw retrieved value cannot serve as an exact key, and
multi-hop needs either a projection that re-lands the retrieval in key space or the
thalamic-relay design (fetch several candidates, gate the chain) — a genuine
architectural addition, not a knob.
