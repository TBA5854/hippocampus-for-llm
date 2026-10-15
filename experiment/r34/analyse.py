"""
r34 analysis — the per-segment profile, compared the way the round pre-registered.

The headline is NOT validation perplexity. The memory-on model has 5.13M
parameters against the control's 4.86M, so a perplexity win is confounded by
capacity and the report already flags that trap.

What cannot be bought with parameters is the SHAPE of the loss across segment
index. `model.forward(idx)` carries no KV cache, so the store is the only channel
between one segment and the next; segment 0 is always read against an empty store
and later segments against a filled one.

One correction, forced by the control's own logs: the control profile is not flat.
Segment 2 is reliably the highest in every eval of `r34_off`, by ~0.02 nats, and
the same bump shows up in `r34_mem`. The model cannot know which segment it is in,
so that is the sampler or the corpus, not memory. Comparing the memory-on profile
against an ASSUMED flat line would read that artifact as signal.

So the statistic is a difference in differences:

    slope(run)  = mean(later segments) - segment 0        # within one model
    effect      = slope(memory) - slope(control)          # artifact cancels

    uv run --group sleep python experiment/r34/analyse.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RUNS = [("r34_off", "control: no memory"),
        ("r34_mem", "memory on"),
        ("r34_shuffle", "control: store scrambled"),
        ("r34_compress", "memory on, 25% write rate"),
        ("r34_off_s1", "control: no memory (seed 1)"),
        ("r34_mem_s1", "memory on (seed 1)"),
        ("r34_off_s2", "control: no memory (seed 2)"),
        ("r34_mem_s2", "memory on (seed 2)"),
        ("r34_off_s3", "control: no memory (seed 3)"),
        ("r34_mem_s3", "memory on (seed 3)"),
        ("r34_mem_misconfigured", "memory on, query/key in DIFFERENT spaces")]
# each memory arm is scored against the control of ITS OWN seed
BASE = {"r34_mem_s1": "r34_off_s1", "r34_mem_s2": "r34_off_s2",
        "r34_mem_s3": "r34_off_s3"}
WINDOW = 5          # average the last N evals, to damp step-to-step jitter


def load(tag):
    p = RESULTS / f"{tag}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    hist = [h for h in d.get("history", []) if h.get("val_per_segment")]
    return d, hist


def profile(hist, window=WINDOW):
    """Mean per-segment loss over the last `window` evals, and its spread."""
    tail = hist[-window:]
    n = len(tail[0]["val_per_segment"])
    mean = [sum(h["val_per_segment"][i] for h in tail) / len(tail)
            for i in range(n)]
    spread = [max(h["val_per_segment"][i] for h in tail)
              - min(h["val_per_segment"][i] for h in tail) for i in range(n)]
    return mean, spread


def slope(mean):
    """Later segments against segment 0. Positive = later segments are EASIER."""
    return mean[0] - sum(mean[1:]) / len(mean[1:])


def main():
    loaded = {}
    for tag, label in RUNS:
        got = load(tag)
        if got is None:
            print(f"{tag:14} -- no result yet")
            continue
        d, hist = got
        if not hist:
            print(f"{tag:14} -- no eval history")
            continue
        loaded[tag] = (d, hist, label)

    if not loaded:
        return

    print(f"{'run':14} {'label':28} {'steps':>6} {'val':>7} {'profile s0..sN':>34} "
          f"{'slope':>8} {'gate':>7}")
    for tag, (d, hist, label) in loaded.items():
        mean, _ = profile(hist)
        last = hist[-1]
        gates = last.get("gates") or {}
        g = max(gates.values()) if gates else float("nan")
        prof = " ".join(f"{v:.3f}" for v in mean)
        print(f"{tag:14} {label:28} {last['step']:>6} {last['val']:>7.4f} "
              f"{prof:>34} {slope(mean):>+8.4f} {g:>7.3f}")

    if "r34_off" in loaded and "r34_mem" in loaded:
        base, _ = profile(loaded["r34_off"][1])
        print("\ndifference in differences (the round's actual measurement)")
        print(f"  control slope           {slope(base):+.4f}")
        for tag in ("r34_mem", "r34_shuffle", "r34_compress",
                    "r34_mem_s1", "r34_mem_s2", "r34_mem_s3",
                    "r34_mem_misconfigured"):
            if tag not in loaded:
                continue
            ref = BASE.get(tag, "r34_off")
            if ref not in loaded:
                continue
            b = profile(loaded[ref][1])[0]
            m, spread = profile(loaded[tag][1])
            eff = slope(m) - slope(b)
            # per-segment gap against the control, which is where a real effect
            # should GROW with segment index rather than sit as a constant offset
            gap = " ".join(f"{bb - v:+.3f}" for v, bb in zip(m, b))
            # The effect statistic alone is ambiguous: it rises when the LATER
            # segments improve AND when SEGMENT 0 degrades. Seed 1 is the case
            # that separates them -- a big effect carried by s0 getting worse.
            # So report both halves, and the net, which is what a user gets.
            later = sum(bb - v for v, bb in zip(m[1:], b[1:])) / (len(m) - 1)
            s0gap = b[0] - m[0]
            net = sum(b) / len(b) - sum(m) / len(m)
            print(f"  {tag:14} effect {eff:+.4f} = later {later:+.4f} "
                  f"- s0 {s0gap:+.4f}   net {net:+.4f}   "
                  f"gap s0..sN {gap}   jitter {max(spread):.3f}")
        print("\n  effect > 0 means later segments got easier BEYOND whatever the\n"
              "  control already does at that segment index. A constant gap across\n"
              "  s0..sN is capacity, not retrieval: segment 0 is read against an\n"
              "  EMPTY store and cannot benefit from memory.")


if __name__ == "__main__":
    main()
