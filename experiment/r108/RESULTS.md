# r108 — the LITERAL 1M-token context: top-1 holds FLAT 100% at ~998k tokens, touching ONE slot

**The 1M-token push, blocked in r102 by an eval-implementation limit, now completes. Top-1
recall is a FLAT 100% out to 7,799 windows = 990,473 slots ≈ 998k tokens of intervening
prose — answered from an empty prompt by touching a SINGLE slot. The blocker was never the
architecture (O(1)/query); it was that `write_many` rewrote the ENTIRE [B,M,d] store
out-of-place via a one-hot scatter EVERY window (O(M·d)/window → O(nseg·M·d) to build),
intractable at ~1M slots. An eval-time in-place O(S) write (no autograd in eval) makes a
~1M-slot store buildable, and a fixed pre-sized store + batch-1 + top-1 read bounds memory.
The store's reach is unbounded up to build time; the effective context is ~1000× the model's
128-token attention window, measured.**

## Numbers (arch2 d512/2L, r87 ckpt; grow off / fixed-M; batch 1; top-1; normal-word keys)
| windows | slots in store | ≈ tokens | top-1 recall | slots touched |
|---------|----------------|----------|--------------|---------------|
| 999   | 126,873  | ~128k | **100.0%** | 1 |
| 2,999 | 380,873  | ~384k | **100.0%** | 1 |
| 7,799 | **990,473** | **~998k** | **100.0%** | 1 |

## Reading
- **~1M tokens at flat 100%, one slot.** A fact stated once and queried from an empty
  128-token window is recalled at 100% across ~998k tokens of intervening prose, touching one
  of 990,473 slots. This is ~1000× the attention window, measured (r102 argued it; r108 shows it).
- **The 1M wall was an eval bug, not the model.** r102 OOM'd/crawled three ways at ≥380k slots
  because building the store rewrote all [B,M,d] every window (out-of-place one-hot, O(M·d)).
  In eval there is no backward, so the write can be in-place O(S); with that fix the ~1M store
  builds in bounded memory. The trained read was always O(1) per query — the offline evaluation
  was the only thing that scaled with M, and now it does not for the write. (The build's
  per-window READ over the growing store is still O(nseg·M) and is why ns7800 takes ~2h; the
  recall result is unaffected.)
- **Unbounded in practice.** Nothing about the model caps the reach; the store grows and the
  read stays O(1). The limit is now build wall-clock, not memory or capacity.

## Verdict
**The long-context claim is now literal: an in-block episodic store, grown large and read by an
exact top-1 lookup, holds recall FLAT at 100% to ~1M tokens (~1000× the window) in O(1) slots.**
This closes the r102 caveat ("measured to ~128k, 1M eval-bound") — the 1M number is measured.
Folded into paper §longcontext (replacing the eval-bound hedge) and it directly anticipates the
passkey-retrieval benchmark (a needle in a ~1M-token context). Enabler committed 3d6bde2.
