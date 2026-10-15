# r31d — results: slot-width compression fails, and the reason is the design's own foundation

## Numbers (1x store, 128 symbols, 8 facts, cold, 8000 steps; `--no-write-identity --query-init key_proj`)

| run | d_mem | write rate | store bytes | **final recall** | slot@1 | read mass | gate |
|---|---|---|---|---|---|---|---|
| baseline (r23) | 256 (raw states) | 100% | 1x | **99.7%** | 100% | 1.00 | 0.16 |
| r31d_w64 | 64 | 100% | 1/4 | **0.5%** | **0.0%** | 0.00 | 0.006 |
| r31d_w32 | 32 | 100% | 1/8 | **1.6%** | **0.0%** | 0.00 | 0.073 |
| r31d_w64_r25 | 64 | 25% | 1/16 | **0.5%** | **0.0%** | 0.00 | 0.002 |

Chance 0.8%. All three are at chance.

## Verdicts

- **H31d-a (d_mem 64 costs little) — falsified.** 0.5%, and not marginally: `slot@1` is **0.0%** from the first eval
  to the last. The read never lands on the right slot at all.
- **H31d-b (32 is where it breaks) — superseded.** It breaks at 64; 32 is no worse because both are already at floor.
- **H31d-c (width and rate compose) — falsified.** 16x on bytes is at chance, as the width factor alone predicts.

## Diagnosis — this is the cost of the thing that makes everything else work

The failure signature is unambiguous and it is **matching**, not content: `slot@1 = 0.0%`, read mass `0.00`,
content-miss `0.0%` (there is no content to miss when nothing is retrieved). Compare every working configuration in
this project, where slot@1 is pinned at 100% from the first eval.

The reason is the property the whole design rests on. A layer-0, position-free key **is the token embedding**, so
matching is near-exact token lookup — that is why r5 got 100% on three seeds, why r26 could decode the store exactly,
and why sleep can replay from keys with no decoder. Push the key through a frozen random projection to 64 dimensions
and that exactness is destroyed: distinct tokens land close together, and with `beta=32` on a near-degenerate
similarity there is no signal to retrieve on. This is Kanerva's correlated-address failure, which the report already
names as the failure this implementation hit once before (381 slots within 0.003 cosine, fixed by centring). Centring
cannot fix a projection that has thrown the distinctions away.

`--query-init key_proj` (added this round) is not the problem: it is the correct narrow-store analogue of the identity
query, copying the write head's frozen key projection so query and key share a space, and a test pins that. The
projection is applied to *both* sides; what fails is that the space itself no longer separates tokens.

## The honest reading

**The two "compression" axes are not interchangeable, and only one of them is available here.**

| axis | result |
|---|---|
| write RATE (r30) | **4x free** (100.0% vs 99.7%), 8x at 88.3% via curriculum |
| slot WIDTH (this round) | **fails at 4x** (chance), for the same reason exact matching works |

A narrower slot is not impossible in principle — it needs keys that stay separable under projection, which means
learned keys rather than raw embeddings, which is the *contextual key* direction OBJECTIVE.md has listed as unbuilt
since r4 (where contextual keys drifted). That is a different project, and it trades away exact decodability, hence
sleep-as-eviction as currently built.

So the plan's original "compression" ask is answered in the negative for this architecture, and in the positive for
the axis the user proposed instead.

## What this does NOT show

- One seed, one store size, one write-identity variant (frozen JL projection). A *learned* compressor, or a
  product-quantised store, is untested and is the obvious alternative if bytes must come down.
- d_mem between 64 and 256 is unexplored — the break is somewhere in that range, and the curve might be steep rather
  than a cliff at 64. Not worth the GPU time unless slot bytes become the binding constraint.
