# r31 — episodes: eviction that protects what can be consolidated

## Ideate

The store has been a ring buffer since r1. That was fine while it never overflowed (r1-r30 all sized the store to hold
everything), but the original ask was explicit:

> "a episode can have multiple token, we should store complete episode alone, so maybe eviction based only replacing
> stale incomplete stuff but not complete episode which is stale"

Two things since then make this concrete rather than a preference:

1. **r26/r27 gave completeness a mechanical meaning.** A complete episode has a replay target, so sleep can consolidate
   it and free its slots; an open one cannot. "Complete vs incomplete" is exactly "consolidatable vs not".
2. **r32 found the repair path needs episode records anyway.** With a content-based write policy, re-encoding after
   sleep recomputes the selection against changed weights and collapses (6.3%). The written slot indices have to travel
   with the episode. An episode table is where they belong.

`ep_id[b, slot]` (-1 = free) plus a completeness flag per episode, and an allocator that spends free slots first, then
incomplete episodes oldest-first, and touches a complete episode only when nothing else is left. Implemented this round
with six tests, including the one that matters: a complete episode survives an overflow that evicts an incomplete one,
where the ring buffer loses it.

## The experiment: make the store overflow

Every previous round sized the store so eviction never happened. Here the store is deliberately too small:
`--memory-size 192` with 4 segments writing ~96 slots each (25% write rate) = ~288 slots wanted, 192 available.

Segment 0 holds the facts and is marked **complete** (an event with its response: the k->v pairs). Distractor segments
are **incomplete**. So the policy has something to protect and something to spend.

## Hypotheses

- **H31a — the ring buffer loses the facts.** With ring eviction the oldest slots go first, and the oldest slots are
  segment 0 — the facts. Predicted: recall collapses toward chance despite a perfectly healthy read.
- **H31b — episode-aware eviction keeps them.** Same store size, same everything: the facts survive because their
  episode is complete, and recall stays high. Falsified if recall is equally bad, which would mean the fact slots are
  lost for some other reason (e.g. the query segment's own writes).
- **H31c — an uncompressed store makes the gap worse**, since 381 slots per segment against 192 means the ring wraps
  within a single segment. Compression and episode eviction should compose: fewer slots per episode means more
  episodes fit before anything has to be evicted at all.

## Runs (1x-sized store deliberately UNDER-sized; 128 symbols, 8 facts, cold, 8000 steps)

| box | name | store | write rate | eviction |
|---|---|---|---|---|
| box1 | r31_ring_m192 | 192 | 25% | ring (current) |
| box1 | r31_ep_m192 | 192 | 25% | **episode-aware** |
| box2 | r31_ring_m192_full | 192 | 100% | ring |
| box2 | r31_ep_m192_full | 192 | 100% | **episode-aware** |
