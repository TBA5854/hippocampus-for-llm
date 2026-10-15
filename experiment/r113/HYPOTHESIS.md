# r113 — Characterize the r112 write-order decay (primacy at scale)

## The r112 surprise
Passkey depth sweep (needle-in-a-haystack, where the fact sits in the context):
- **ctx64 (8k tok)**: recall 100/100/100/87.5/75% across depth 0→1.
- **ctx512 (64k tok)**: recall **100/62.5/50/25/12.5%** — steep, monotonic.
- shuffle control: **0%** everywhere → the decay is genuine retrieval degrading.

So there is **no lost-in-the-middle** (the classic attention failure), but a strong
**primacy / write-order effect**: a fact stated **early** persists (100%); a fact
stated **late into a saturated store** is nearly lost (12.5%). The penalty **grows
with store saturation** (mild at 8k, severe at 64k). This is the *opposite* of a KV
cache's recency, and likely the same limit as the 128-fact capacity frontier viewed
along the time axis.

## What r113 tests (behavioral, no internals)
Three candidate causes, each with a clean behavioral discriminator on the r87/d512 ckpt:

1. **Store saturation** (slots/capacity). Sweep ctx = 32 / 128 / 512 windows. If the
   decay **steepens with ctx**, it is saturation-driven (predicted).
2. **Novelty write-gate** (`write_gate=True, write_threshold=0.6`): a late needle in a
   full store may read as non-novel and never get written. Discriminator: **`--rare-keys`**
   (unique, always-novel keys). If the decay **flattens** with rare keys, it's the gate.
3. **Ring lapping**: ruled out by slot counts (65k < 77k M), but confirm with
   **`--store-grow`** (never laps). If grow **does not change** the decay, lapping is out.

All arms: top-1 read, batch 2, **iters 8 (n=16, de-noised** vs r112's n=8), depth grid
{0,.25,.5,.75,1} (fine {0,.1,.25,.5,.75,.9,1} at ctx512).

## Reads
- Decay steepens with ctx → **saturation** is the axis.
- rare-keys flattens it → **write-gate** suppression of late facts (a fixable policy).
- rare-keys does *not* flatten → points to **key-centering drift** (`center_keys`), a
  representational limit (leave mechanism as hypothesis; no mech-interp deep-dive).

## Paper impact
Turns the passkey benchmark from "flat/no-LitM" into a sharper, more honest result:
**content-addressed memory has no middle penalty but a write-order/saturation limit** —
early facts are privileged, and it connects the long-context reach to the capacity
frontier. Written up as measured; not re-tuned to look flat.
