# r51 — close the objective on real language: persistence + exceeding the context window

The objective's "Done means": cross-segment recall out of context, **store persisted to disk and
reloaded**, with matched controls at chance. r49/r50 nailed the recall (100% through 32 facts, 99% at
64) and the scrambled control (chance). Two things remain to *close* it on real language, both asked
for explicitly:

1. **Persistence across a process death** (done on MQAR in r6, never on BPE facts).
2. **Match/exceed a model's context window** — show the store gives the model an *effective* context
   many times its literal attention window, and state the general recipe for storing more.

Both are **offline** probes on the already-trained 100%/99% checkpoints. No training this round.

## The context-window framing (why this is a real claim, not a relabel)

Segments do **not** cross-attend — `model.forward` is stateless per segment, so the store is the ONLY
channel between them. The model's literal attention window is `seg_len = 128` tokens. In the task the
facts are all stated in **segment 0** (one 128-token window); the query is a **separate** window
(`facts_in_context=False`) containing none of them. So the model already answers about facts outside
its current window — the store bridges the gap. `(n_seg - 1)` counts the windows of unrelated prose
between the facts and the query.

## H51a — persistence

**Claim:** a store written by one process, saved, and reloaded in a FRESH process answers as well as
in-process; a mismatched store file falls to chance.
**Predict:** `reloaded_recall ≈ in_process_recall` (both ~100% on the 32-fact model); `mismatched ≈
chance`. **Falsified if** reloaded recall drops materially below in-process (state leaked through the
interpreter, not the file) or mismatched stays high (answers not actually coming from the file).

    uv run python -m mech.persist write --run r49_bpe32_curr.json --ckpt r49_bpe32_curr.pt --dir D
    uv run python -m mech.persist read  --run ...              --ckpt ...              --dir D
    uv run python -m mech.persist read  --run ... --ckpt ... --dir D --mismatch

## H51b — how many context windows does the store bridge, and what sets the limit

**Claim:** recall from the empty query window holds while the store has room for seg-0's facts plus the
intervening prose (~`n_seg × (seg_len-1)` slots), and collapses once later prose evicts the facts. The
collapse point moves out **linearly with store size M** — so "store more" = "grow M".
**Predict:** at `M=512` the bridge holds to ~4 windows (508 slots ÷ 127/seg) then drops; doubling M
roughly doubles the reach. **Falsified if** recall is flat in `n_seg` (store never the limit — then M is
not the lever) or collapses well before the slot budget is spent (something other than capacity binds).

    uv run python -m mech.persist context --run r50_bpe64.json --ckpt r50_bpe64.pt \
        --mem-list 512 1024 2048 4096 --nseg-list 4 8 16 32 --out r51_context.json

64 facts is itself already past the window: listed as `key value` pairs they need ~128+ tokens, yet the
model answers them from a separate empty 128-token prompt. The sweep quantifies how much *further* — how
many windows of separation — the store reaches, and shows M is the knob.

## General recipe (the write-up target)

Two independent capacity axes, to be stated plainly in RESULTS + OBJECTIVE:
- **Span / tokens bridged** scales LINEARLY with store size **M** (this round measures the slope).
- **Distinct facts** is curriculum-bound (bootstrapping, r25/r49), needs `M ≥ facts × slots-per-fact`
  but is not M-limited below that (r39). Single-token keys (BPE) remove the multi-token addressing tax
  (r36); the curriculum ladder escapes the fact-count deadlock.
