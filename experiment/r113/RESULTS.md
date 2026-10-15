# r113 — RESULTS: what causes the r112 write-order decay (behavioral decomposition)

**Verdict: the write-order decay is (1) driven by store SATURATION, (2) NOT ring
lapping, and (3) PARTLY the novelty write-gate — late common-token needles read as
non-novel and get gated out — with a residual second factor (key-quality / centering)
that unique keys do not fully remove.** All behavioral (config/flag ablations), no
internals.

## Setup
`mech.persist passkey`, top-1, batch 2, **iters 8 (n=16, de-noised)**, on d512/r87 (box1,
full) and d256/r92 (box2, cross-model). Depth grids coarse {0,.25,.5,.75,1} and fine
{0,.1,.25,.5,.75,.9,1} at ctx512.

## (1) Saturation drives it — d512 depth sweep at growing ctx
| ctx (slots) | d0.0 | d0.25 | d0.5 | d0.75 | d1.0 |
|---|---|---|---|---|---|
| 32 (4,064)  | 100% | 100% | 87.5% | 81% | 81% |
| 128 (16,256) | 100% | 87.5% | 68.8% | 56.2% | 50% |
| 512 (65,024) | 100% | 56.2% | 31.2% | 12.5% | 6.2% |

The late-needle (depth 1.0) recall falls **81% → 50% → 6.2%** as the store grows
4k → 16k → 65k slots. The decay **steepens monotonically with saturation** — it is a
function of how full the store is, not of absolute context length per se.

## (2) NOT ring lapping — grow == baseline
`--store-grow` (append, never lap) at ctx512: **100/75/56.2/31.2/12.5/6.2/6.2** —
*identical* to the fixed-ring baseline (100/75/56.2/31.2/12.5/6.2/6.2). Since no slot is
ever overwritten and the curve is unchanged, the decay is **not** the ring buffer
lapping. (Consistent with slot counts: 65,024 < M=76,690, no lap.)

## (3) Partly the novelty write-gate — rare-keys lifts the tail (d512, ctx512)
| depth | 0.0 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|
| baseline (common keys) | 100% | 75% | 56.2% | 31.2% | 12.5% | 6.2% | 6.2% |
| `--rare-keys` (unique) | 75% | 62.5% | 37.5% | 37.5% | 31.2% | 25% | **25%** |

Unique, always-novel keys **raise the late-needle tail 4×** (depth 1.0: 6.2% → 25%;
the curve flattens). So a late **common-token** needle — whose key token recurs in the
intervening prose — reads as *non-novel* and is suppressed by the `write_threshold=0.6`
novelty gate; a **unique** key stays novel and gets written. **The novelty write-gate is
part of the cause, and it is a fixable policy** (lower/disable the gate, or key on a
salient token).

**But the rescue is only partial** and rare keys *lower* the early end (depth 0: 100% →
75%, the r102 key-quality effect of under-trained rare embeddings). Even with unique
keys the curve still decays (75% → 25%), so a **second, residual write-order factor**
remains — most likely key-centering drift (`center_keys` EMA shifts toward prose as the
store fills) or slot accumulation. Left as a hypothesis; not chased into internals
(mech-interp deferred).

## Cross-model (d256, box2)
| ctx | d0.0 | d0.25 | d0.5 | d0.75 | d1.0 |
|---|---|---|---|---|---|
| 32  | 87.5% | 12.5% | 12.5% | 12.5% | 0% |
| 128 | 75% | 12.5% | 6.2% | 6.2% | 6.2% |
| 512 (fine) | 81.2% | 0% (all deeper) | | | |

On the weaker model the decay is **near-binary** from any saturation — only the
first-written fact survives — consistent with saturation reached far sooner on a
lower-capacity store. The mechanism arms complete the cross-model picture at ctx512:

| d256 ctx512 arm | d0.0 | d0.1 | d0.25 | d0.5 | d0.75 | d0.9 | d1.0 |
|---|---|---|---|---|---|---|---|
| baseline | 81.2% | 0 | 0 | 0 | 0 | 0 | 0 |
| `--rare-keys` | 31.2% | 0 | 0 | 0 | 0 | 0 | 0 |
| `--store-grow` | 81.2% | 0 | 0 | 0 | 0 | 0 | 0 |

Two of the three findings hold cross-model; the third does not:
- **Not lapping — holds.** `--store-grow` is byte-identical to the fixed-ring baseline
  (81.2/0…), exactly as on d512.
- **Saturation — holds.** The collapse is near-binary at every ctx (d256 saturates far
  sooner than d512).
- **Write-gate rescue — does NOT transfer.** On d512 unique keys lifted the late tail
  4× (6→25%); on d256 `--rare-keys` leaves the tail at **0%** and *lowers* the early end
  hard (81→31%). d256's rarest-token embeddings are too under-trained to serve as keys at
  all (the r102 key-quality effect, severe on the weak model), so the fixable write-gate
  policy only helps where the embeddings can support a unique key.

**Verdict: the not-lapping and saturation findings are cross-model; the write-gate
rescue is model-dependent** — it works on the strong model, where rare-token embeddings
are good enough to key, and not on the weak one, where key quality dominates.

## Paper impact
Folded into §longcontext (passkey paragraph) and §limitations: content-addressed memory
has **no lost-in-the-middle but a write-order/saturation limit**; it is partly the novelty
write-gate (fixable) and ties the long-context reach to the capacity frontier. Reported
as measured — not re-tuned to look flat.
