# r101 — growable store + top-1 sparse read = an UNBOUNDED store read at FLAT recall

**Grow (r100, no ring lapping) and the top-1 sparse read (r62, immune to distractor mass)
compose: top-1 holds 100% at every bridge distance out to 199 windows (25,273 slots),
touching ONE slot per query, while the dense read decays 85→60% over the same span. So the
store is unbounded (append like a KV cache) AND read at flat 100% in O(1) slots.**

## Context-bridge recall vs distance (arch2_d512_L2, --store-grow)
| windows (slots) | top-1 (1 slot) | dense (all slots) |
|-----------------|----------------|-------------------|
| 15 (1905)   | **100.0%** | 85.4% |
| 63 (8001)   | **100.0%** | 75.0% |
| 127 (16129) | **100.0%** | 64.6% |
| 199 (25273) | **100.0%** | 60.4% |

## Reading
- **The two mechanisms compose exactly.** r100 showed growth removes the ring cliff but the
  dense read still decays with size (distractor mass). r62 showed top-1 is immune to that
  decay. Together: the growable store holds every fact (no lap), and top-1 reads the one
  matching slot regardless of how large the store gets — **flat 100% out to 199 windows,
  1 slot touched**, from a 7.8M-param model.
- The dense column is the control: it decays 85→60% as the store grows to 25k slots, which
  is precisely the distractor dilution top-1 avoids.

## Verdict
Effective context is now bounded by neither the attention window (this is 199× it) nor the
store capacity (grow removes the cap) nor the read cost (O(1) slots): it is bounded only by
key collisions (r52/r89), which r102 pushes toward 1M tokens to locate. The store is a
stateless, exact, unbounded, O(1)-read memory. Folded into the paper long-context section.
