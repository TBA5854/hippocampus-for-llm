"""
tasks.py — synthetic cross-segment recall.

P2 showed why this is needed: on character-level Shakespeare the per-segment
loss profile is flat (s0 2.365 / s3 2.366), because predicting the next
character never depends on a specific fact from three segments ago. There is
nothing for a persistent store to win, so an LM corpus alone cannot test the
claim.

This task creates that demand directly:

    segment 0        k1:v1 k2:v2 ... kn:vn        <- the facts
    segments 1..N-2  Shakespeare                  <- distractors, and the LM signal
    segment N-1      ...........?kj:              <- the query; answer is vj

Each segment is fed on its own, so by the time the query arrives segment 0 is
long out of context. The answer is reachable only through the store. Three
conditions bracket the result:

    hc off            floor    — cannot possibly answer above chance
    hc on             the number
    facts in context  ceiling  — facts pasted into the query segment
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
TASK_CHARS = ALPHABET + ":?. "


def make_batch(ids, tok, batch: int, n_seg: int, seg_len: int, n_facts: int,
               device, generator=None, facts_in_context: bool = False):
    """-> x, y [B, n_seg, seg_len], answer_mask [B, n_seg, seg_len] bool."""
    enc, pad = tok.encode, tok.stoi.get(" ", 0)
    x = torch.full((batch, n_seg, seg_len), pad, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    g = generator
    for b in range(batch):
        perm = torch.randperm(len(ALPHABET), generator=g)
        letters = [ALPHABET[i] for i in perm.tolist()]
        keys = [letters[2 * i] + letters[2 * i + 1] for i in range(n_facts)]
        vperm = torch.randperm(len(ALPHABET), generator=g)
        vl = [ALPHABET[i] for i in vperm.tolist()]
        vals = [vl[2 * i] + vl[2 * i + 1] for i in range(n_facts)]
        facts = " ".join(f"{k}:{v}" for k, v in zip(keys, vals))

        j = int(torch.randint(n_facts, (1,), generator=g))
        query = f"?{keys[j]}:{vals[j]}"

        # segment 0 — the facts, TILED to fill the segment. The write head mean-
        # pools the whole segment into one vector, so leaving the tail as padding
        # would dilute the facts with blanks and confound "the store cannot carry
        # this" with "the summary was mostly spaces".
        unit = facts + " . "
        row = enc(unit * (seg_len // max(len(unit), 1) + 1))[:seg_len]
        x[b, 0, :len(row)] = torch.tensor(row)

        # middle segments — distractor prose, which is also the LM signal
        for s in range(1, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        # final segment — query, right-aligned so the answer ends the segment
        tail = enc(query)
        if facts_in_context:                      # ceiling condition
            tail = enc(facts + " " + query)
        tail = tail[-seg_len:]
        off = seg_len - len(tail)
        x[b, n_seg - 1, off:] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - len(enc(vals[j])):] = True

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad
    # answer_mask marks TARGET positions, so shift it to line up with y
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


@torch.no_grad()
def exact_match(logits, y, mask) -> tuple[int, int]:
    """Fraction of examples whose every answer character is correct."""
    pred = logits.argmax(-1)
    ok = ((pred == y) | ~mask)
    per_example = ok.reshape(mask.shape[0], -1).all(-1)
    has = mask.reshape(mask.shape[0], -1).any(-1)
    return int((per_example & has).sum()), int(has.sum())


@torch.no_grad()
def answer_stats(logits, y, mask, out: dict) -> None:
    """Accumulate PER-CHARACTER accuracy and answer-position CE into `out`.

    This exists because exact match is a blunt instrument at character level,
    and the project has already been misled by it once. The note that created
    MQAR says so directly: the answer is two characters, so a real retrieval
    edge (CE 3.148 against a 3.256 scrambled control, chance log(26) = 3.258)
    "almost never produced a correct pair". Exact match read ~0 while the model
    genuinely knew something.

    So a character-level recall round reports three things, and they fail
    independently:

      * `exact`     — both characters right. The capability claim.
      * `char_acc`  — per character, chance 1/26 = 3.8%. Catches a half-learned
                      binding that exact match rounds to zero.
      * `answer_ce` — cross entropy on the answer positions, against log(26) =
                      3.258 for a model that knows nothing. Catches an edge too
                      small to change any argmax at all.

    A claim needs `exact`; a NEGATIVE needs all three, or it cannot distinguish
    "the store does not carry this" from "the ruler is too coarse to see it".
    """
    pred = logits.argmax(-1)
    sel = mask
    n = int(sel.sum())
    if n == 0:
        return
    ok = (pred == y) & sel
    out["char_hits"] = out.get("char_hits", 0) + int(ok.sum())
    out["char_total"] = out.get("char_total", 0) + n

    # Per-position accuracy WITHIN the answer (r36). r35 measured 47.1% of
    # characters right but only 10.0% of pairs, so the two characters do not
    # fail independently (0.471^2 = 22% would be independent). Splitting them
    # says which one the store loses.
    #
    # It is a real question because the two characters come from DIFFERENT
    # slots: with value_shift 1, the slot keyed on ":" returns the first answer
    # character and the slot keyed on that character returns the second. So
    # "first right, second wrong" is one more hop failing, while "both wrong
    # together" points at the match rather than the content.
    rank = sel.cumsum(dim=1) - 1                       # 0,1,... within the answer
    for k in range(int(rank[sel].max()) + 1 if n else 0):
        at = sel & (rank == k)
        t = int(at.sum())
        if t:
            out[f"pos{k}_hits"] = out.get(f"pos{k}_hits", 0) + int((ok & at).sum())
            out[f"pos{k}_total"] = out.get(f"pos{k}_total", 0) + t
    lp = F.log_softmax(logits.float(), dim=-1)
    nll = -lp.gather(2, y.clamp_min(0)[..., None]).squeeze(2)
    out["ce_sum"] = out.get("ce_sum", 0.0) + float((nll * sel).sum())
    out["ce_n"] = out.get("ce_n", 0) + n


# --------------------------------------------------------------- persona ----
# Consolidation needs facts that RECUR. make_batch() above draws fresh random
# pairs every episode, so sleeping on it would be memorising noise — there is
# nothing stable for the weights to absorb. A persona is a fixed fact set that
# many episodes reference, which is what the 15 personal facts were in v0.2.

class Persona:
    """A fixed set of key:value facts belonging to one 'user'."""

    def __init__(self, n_facts: int, generator=None):
        perm = torch.randperm(len(ALPHABET), generator=generator).tolist()
        vperm = torch.randperm(len(ALPHABET), generator=generator).tolist()
        self.keys = [ALPHABET[perm[2 * i]] + ALPHABET[perm[2 * i + 1]]
                     for i in range(n_facts)]
        self.vals = [ALPHABET[vperm[2 * i]] + ALPHABET[vperm[2 * i + 1]]
                     for i in range(n_facts)]

    @property
    def facts(self) -> str:
        return " ".join(f"{k}:{v}" for k, v in zip(self.keys, self.vals))

    def query(self, j: int) -> str:
        return f"?{self.keys[j]}:{self.vals[j]}"


def persona_batch(ids, tok, persona: Persona, batch: int, n_seg: int,
                  seg_len: int, device, generator=None,
                  facts_in_context: bool = False, no_facts: bool = False):
    """Episodes about ONE persona.

    no_facts=True drops segment 0 entirely (distractors only), which is the
    post-sleep probe: nothing in context and nothing in the store, so a correct
    answer can only have come from the weights.
    """
    enc, pad = tok.encode, tok.stoi.get(" ", 0)
    x = torch.full((batch, n_seg, seg_len), pad, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)
    g = generator

    for b in range(batch):
        j = int(torch.randint(len(persona.keys), (1,), generator=g))
        query = persona.query(j)

        if not no_facts:
            unit = persona.facts + " . "
            row = enc(unit * (seg_len // max(len(unit), 1) + 1))[:seg_len]
            x[b, 0, :len(row)] = torch.tensor(row)
        else:
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, 0] = ids[start:start + seg_len]

        for s in range(1, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        tail = enc(persona.facts + " " + query) if facts_in_context else enc(query)
        tail = tail[-seg_len:]
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - len(enc(persona.vals[j])):] = True

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# ------------------------------------------------------------------ MQAR ----
# Character-level made exact-match a poor instrument: the answer was TWO
# characters, so a real but small retrieval edge (measured: CE 3.148 vs a 3.256
# scrambled control, log(26)=3.258) almost never produced a correct pair.
# Token-level symbols make the answer exactly ONE token, so accuracy measures
# the binding directly. This is multi-query associative recall, the standard
# synthetic probe for "can this architecture store and retrieve key->value".

N_SYMBOLS = 32          # distinct keys, and distinct values; chance = 1/32 = 3.1%


def mqar_vocab(base_vocab: int, n_symbols: int = N_SYMBOLS) -> int:
    """base chars + N key symbols + N value symbols + one query marker."""
    return base_vocab + 2 * n_symbols + 1


def _sym(base_vocab: int, n_symbols: int = N_SYMBOLS):
    keys = base_vocab
    vals = base_vocab + n_symbols
    query = base_vocab + 2 * n_symbols
    return keys, vals, query


def mqar_batch(ids, base_vocab: int, batch: int, n_seg: int, seg_len: int,
               n_facts: int, device, generator=None,
               facts_in_context: bool = False, no_facts: bool = False,
               tile: bool = True, n_symbols: int = N_SYMBOLS):
    """-> x, y [B, n_seg, seg_len], answer_mask [B, n_seg, seg_len].

    seg 0        k1 v1 k2 v2 ... (tiled)
    seg 1..N-2   distractor text
    seg N-1      ..... <query> kj     with target vj at the next position

    tile=False writes each pair ONCE at the start of segment 0 and fills the
    rest with distractor text. Tiling lets induction make v_j predictable from
    earlier repeats, so a store that holds the model's PREDICTION at a key can
    look like it stores facts; r6 measured exactly that dependency. A single
    occurrence cannot be predicted from context, so only a store that holds
    what was OBSERVED can answer it.
    """
    kbase, vbase, qtok = _sym(base_vocab, n_symbols)
    g = generator
    x = torch.zeros(batch, n_seg, seg_len, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    for b in range(batch):
        ks = torch.randperm(n_symbols, generator=g)[:n_facts]
        vs = torch.randperm(n_symbols, generator=g)[:n_facts]
        pairs = []
        for k, v in zip(ks.tolist(), vs.tolist()):
            pairs += [kbase + k, vbase + v]

        if no_facts:
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, 0] = ids[start:start + seg_len]
        elif tile:
            reps = seg_len // max(len(pairs), 1) + 1
            x[b, 0] = torch.tensor((pairs * reps)[:seg_len])
        else:
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, 0] = ids[start:start + seg_len]
            x[b, 0, :len(pairs)] = torch.tensor(pairs[:seg_len])

        for s in range(1, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        j = int(torch.randint(n_facts, (1,), generator=g))
        tail = [qtok, kbase + int(ks[j]), vbase + int(vs[j])]
        if facts_in_context:
            tail = pairs + tail
        tail = tail[-seg_len:]
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - 1] = True          # the value token itself

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = 0
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]                       # shift onto target space
    return x.to(device), y.to(device), m.to(device)


# ---------------------------------------------------------- BPE token-level --
# r36 showed the char-level ceiling is the multi-token-key tax. This is the same
# recall task as make_batch, but keys and values are SINGLE real word-tokens from
# a BPE vocabulary, and the distractor is BPE-encoded prose. So a key is one
# token (addressable, like MQAR) yet a real word competing with the prose vocab
# (unlike MQAR's synthetic out-of-vocab symbols). value_shift 1 makes the slot
# keyed on k_j return v_j in a single hop — no delimiter, no 1/n_facts draw.

def bpe_recall_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                     key_pool, val_pool, qtok_id: int, pad_id: int, device,
                     generator=None, facts_in_context: bool = False,
                     tile: bool = True, needle_seg: int = 0):
    """-> x, y [B, n_seg, seg_len], answer_mask. Single-token word keys/values.

        seg d        k1 v1 k2 v2 ... (tiled)   <- the needle, at depth d = needle_seg
        others 0..N-2 BPE prose distractors
        seg N-1      ... <q> kj      target vj at the next position

    needle_seg (default 0) is the passkey "depth": which context window holds the
    facts. 0 reproduces the historical seg-0 placement exactly (byte-identical RNG).
    """
    g = generator
    key_pool = torch.as_tensor(key_pool)
    val_pool = torch.as_tensor(val_pool)
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    for b in range(batch):
        ks = key_pool[torch.randperm(len(key_pool), generator=g)[:n_facts]]
        vs = val_pool[torch.randperm(len(val_pool), generator=g)[:n_facts]]
        pairs = []
        for k, v in zip(ks.tolist(), vs.tolist()):
            pairs += [k, v]

        d = min(max(int(needle_seg), 0), n_seg - 2)    # needle window (clamp to a context seg)
        for s in range(n_seg - 1):                      # all context windows
            if s == d:                                  # the needle: fact pairs
                if tile:
                    reps = seg_len // max(len(pairs), 1) + 1
                    x[b, s] = torch.tensor((pairs * reps)[:seg_len])
                else:
                    start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
                    x[b, s] = ids[start:start + seg_len]
                    x[b, s, :len(pairs)] = torch.tensor(pairs[:seg_len])
            else:                                       # distractor prose
                start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
                x[b, s] = ids[start:start + seg_len]

        j = int(torch.randint(n_facts, (1,), generator=g))
        tail = [qtok_id, int(ks[j]), int(vs[j])]
        if facts_in_context:
            tail = pairs + tail
        tail = tail[-seg_len:]
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - 1] = True          # the value token

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# r89 — COLLISION. C facts SHARE each key token, distinguished only by a CONTEXT
# token: fact = `c k v` (context, shared key, value); value_shift keys on k and
# returns v. With an exact single-token key the C slots keyed by k collide, so a
# query lands on 1/C of them (the r36 draw). The fine (bigram) key emb(k)+a*emb(c)
# separates them; the query `<q> c k` has the same (c,k) adjacency so it reproduces
# the composite. `scramble_context` swaps in the WRONG context at query (control:
# must fail). Tests dentate-gyrus pattern separation against a real collision axis.
def bpe_collision_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                        key_pool, val_pool, qtok_id: int, pad_id: int, device,
                        generator=None, collisions_per_key: int = 4,
                        scramble_context: bool = False):
    """-> x, y, m. C facts per key token; query (context, key) -> its value."""
    g = generator
    key_pool = torch.as_tensor(key_pool)
    val_pool = torch.as_tensor(val_pool)
    C = max(1, collisions_per_key)
    K = max(1, n_facts // C)                     # distinct (colliding) key tokens
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)
    for b in range(batch):
        kperm = key_pool[torch.randperm(len(key_pool), generator=g)]
        keys = kperm[:K]
        ctx = kperm[K:K + K * C]                 # distinct context tokens (disjoint from keys)
        vals = val_pool[torch.randperm(len(val_pool), generator=g)[:K * C]]
        triples = []                             # (context, shared-key, value)
        t = 0
        for ki in range(K):
            for _ in range(C):
                triples.append((int(ctx[t]), int(keys[ki]), int(vals[t]))); t += 1
        flat = []
        for c, k, v in triples:
            flat += [c, k, v]
        reps = seg_len // max(len(flat), 1) + 1
        x[b, 0] = torch.tensor((flat * reps)[:seg_len])
        for s in range(1, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]
        j = int(torch.randint(len(triples), (1,), generator=g))
        c, k, v = triples[j]
        cq = c
        if scramble_context and len(triples) > 1:     # control: a WRONG context token
            jj = (j + 1 + int(torch.randint(len(triples) - 1, (1,), generator=g))) % len(triples)
            cq = triples[jj][0]
        tail = [qtok_id, cq, k, v][-seg_len:]         # <q> c k  (target v)
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - 1] = True         # the value token
    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# r68 — MULTI-HOP. A chain A->B, B->C (, C->D) is stated once in seg0; the query
# asks for the HEAD (A) and the target is the TAIL (C). Answering needs chaining:
# retrieve B from A, then C from B — one hop per read layer against the ONE shared
# store. Each fact is written as [key, value, pad]; value_shift makes the [key,
# value] adjacency the intended slot. An intermediate token (a value once, a key
# once) gets a spurious value->pad slot that competes with its real key->next
# slot, so chain facts are placed IN ORDER (each after the one feeding it) and the
# recency read (r67) prefers the newer, real slot. `broken` cuts the last link so
# the head cannot reach the tail (control: must fail).
def bpe_hop_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                  key_pool, val_pool, qtok_id: int, pad_id: int, device,
                  generator=None, n_hops: int = 2, broken: bool = False,
                  n_chains: int = 1):
    """-> x, y, m. Query a chain head; target its tail (n_hops away).

    n_chains == 1: a single chain plus distractor pairs (the original form; note it
    LEAKS — the tail is always the last-written value, so "return the most-recent
    value" scores high without chaining; see r70/r71).
    n_chains > 1 (LEAK-PROOF, r72): build several independent chains, shuffle ALL
    their facts into seg0, and query a RANDOM chain's head. "Return the last value"
    now yields some other chain's tail (chance); only genuine chaining from the
    queried head reaches the right tail."""
    assert n_seg >= 3, "hop task needs seg0 (facts), prose, a query seg"
    assert n_facts >= n_hops, "need at least n_hops facts for the chain"
    g = generator
    pool = torch.cat([torch.as_tensor(key_pool), torch.as_tensor(val_pool)])
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    for b in range(batch):
        if n_chains > 1:
            # leak-proof: several disjoint chains, all facts shuffled, query random one
            need = n_chains * (n_hops + 1) + 1
            perm = torch.randperm(len(pool), generator=g)
            toks = pool[perm[:need]].tolist()
            chains = [toks[c * (n_hops + 1):(c + 1) * (n_hops + 1)]
                      for c in range(n_chains)]
            qc = int(torch.randint(n_chains, (1,), generator=g))     # queried chain
            facts = []
            for c, ch in enumerate(chains):
                cf = [(ch[i], ch[i + 1]) for i in range(n_hops)]
                if broken and c == qc:               # cut only the queried chain
                    cf[-1] = (toks[-1], ch[-1])
                facts += cf
            order = torch.randperm(len(facts), generator=g).tolist()
            seq = []
            for j in order:
                k, v = facts[j]
                seq += [k, v, pad_id]
            head, target = chains[qc][0], chains[qc][-1]
        else:
            perm = torch.randperm(len(pool), generator=g)
            need = (n_hops + 1) + 2 * (n_facts - n_hops) + 1
            toks = pool[perm[:need]].tolist()
            chain = toks[:n_hops + 1]
            rest = toks[n_hops + 1:]
            chain_facts = [(chain[i], chain[i + 1]) for i in range(n_hops)]
            if broken:
                chain_facts[-1] = (rest.pop(), chain[-1])
            distractors = []
            for i in range(0, len(rest) - 1, 2):
                distractors.append((rest[i], rest[i + 1]))
            seq = []
            for k, v in distractors + chain_facts:   # chain last = newest (recency)
                seq += [k, v, pad_id]
            head, target = chain[0], chain[-1]
        seq = seq[:seg_len]
        x[b, 0, :len(seq)] = torch.tensor(seq)

        for s in range(1, n_seg - 1):                # prose distractor segments
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        tail = [qtok_id, head, target][-seg_len:]    # query head -> target tail
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - 1] = True

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# r81 — CoT-over-memory (route A). Same leak-proof chains in seg0, but the last
# segment writes the chain EXPLICITLY as query/answer blocks: [q t0 t1] [q t1 t2]...
# so the model emits each intermediate then the next, each a clean single-hop store
# lookup (query t_i -> retrieve t_{i+1}). Teacher-forced here; the honest
# autoregressive eval (feed the model's own t1 into the second block) is a separate
# probe. answer positions mark each t_i (predict t_{i+1}).
def bpe_chain_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                    key_pool, val_pool, qtok_id: int, pad_id: int, device,
                    generator=None, n_hops: int = 2, broken: bool = False,
                    n_chains: int = 4):
    """-> x, y, m. Explicit chain-of-thought: query head, emit each intermediate
    then the tail, one query/answer block per hop."""
    assert n_seg >= 3 and n_chains >= 1 and n_facts >= n_hops
    g = generator
    pool = torch.cat([torch.as_tensor(key_pool), torch.as_tensor(val_pool)])
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    for b in range(batch):
        need = n_chains * (n_hops + 1) + 1
        perm = torch.randperm(len(pool), generator=g)
        toks = pool[perm[:need]].tolist()
        chains = [toks[c * (n_hops + 1):(c + 1) * (n_hops + 1)] for c in range(n_chains)]
        qc = int(torch.randint(n_chains, (1,), generator=g))
        facts = []
        for c, ch in enumerate(chains):
            cf = [(ch[i], ch[i + 1]) for i in range(n_hops)]
            if broken and c == qc:
                cf[-1] = (toks[-1], ch[-1])          # cut the queried chain's last link
            facts += cf
        order = torch.randperm(len(facts), generator=g).tolist()
        seq = []
        for j in order:
            k, v = facts[j]
            seq += [k, v, pad_id]
        seq = seq[:seg_len]
        x[b, 0, :len(seq)] = torch.tensor(seq)

        for s in range(1, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        ch = chains[qc]
        tail, ans_idx = [], []
        for i in range(n_hops):                       # [q t_i t_{i+1}] per hop
            # mark the ANSWER token t_{i+1} (block offset 2); the shifted mask then
            # scores the query position t_i predicting it. (Marking t_i instead made
            # the eval score a copy/guess, not retrieval — the r81 bug.)
            ans_idx.append(len(tail) + 2)
            tail += [qtok_id, ch[i], ch[i + 1]]
        tail = tail[-seg_len:]
        off = seg_len - len(tail)
        x[b, n_seg - 1, off:] = torch.tensor(tail)
        for ai in ans_idx:
            pos = off + ai
            if 0 <= pos < seg_len:
                ans[b, n_seg - 1, pos] = True

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# r99 — PARAMETRIC-SEMANTIC CoT. A FIXED synonym->canonical lexicon (the same across every
# example, so the model must learn it in its WEIGHTS, not read it from the store) plus a
# per-example canonical->value fact stored in seg 0. The query is the SYNONYM; the CoT is
# [q syn canon][q canon val] — hop 1 (syn->canon) is served by parametric memory, hop 2
# (canon->val) by the store. Tests whether a CoT hop can be served by weights to bridge a
# semantic cue to the exact store (the store itself can't fuzzy-match synonyms, r98). broken
# = re-pair syn->canon randomly per example (unlearnable) so hop 1 must fail.
def bpe_semchain_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                       lexicon, val_pool, qtok_id: int, pad_id: int, device,
                       generator=None, broken: bool = False):
    """-> x, y, m. Fixed syn->canon lexicon (parametric hop) + stored canon->val (store hop)."""
    g = generator
    lex = list(lexicon)                                    # [(syn, canon), ...] FIXED
    canons = [c for _, c in lex]
    val_pool = torch.as_tensor(val_pool)
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)
    for b in range(batch):
        idxs = torch.randperm(len(lex), generator=g)[:n_facts].tolist()
        pairs = [lex[i] for i in idxs]                     # (syn, canon) from the fixed map
        if broken:                                         # control: random syn per canon (unlearnable)
            rs = torch.randperm(len(lex), generator=g)[:n_facts].tolist()
            pairs = [(lex[rs[k]][0], pairs[k][1]) for k in range(n_facts)]
        vals = val_pool[torch.randperm(len(val_pool), generator=g)[:n_facts]].tolist()
        seq = []
        for (syn, canon), v in zip(pairs, vals):
            seq += [canon, v, pad_id]                      # store canon->val ONLY (not syn->canon)
        seq = seq[:seg_len]
        x[b, 0, :len(seq)] = torch.tensor(seq)
        for s in range(1, n_seg - 1):
            st = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[st:st + seg_len]
        j = int(torch.randint(n_facts, (1,), generator=g))
        (syn, canon), v = pairs[j], vals[j]
        tail = [qtok_id, syn, canon, qtok_id, canon, v]    # [q syn canon][q canon val]
        off = seg_len - len(tail)
        x[b, n_seg - 1, off:] = torch.tensor(tail)
        ans[b, n_seg - 1, off + 2] = True                  # canon (hop1: parametric syn->canon)
        ans[b, n_seg - 1, off + 5] = True                  # val   (hop2: store canon->val)
    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)


# r67 — UPDATE / recency. A fact is stated once in seg 0, then ONE fact is
# re-stated with a new value in a later segment; the query asks for that key and
# the target is the NEW value. Facts are stated once (no tiling) so the store's
# global write order cleanly separates the stale slot (early) from the update
# (late) — with tiling the old value is written right up to the segment boundary
# and is almost as recent as the update, so recency cannot tell them apart.
def bpe_update_batch(ids, batch: int, n_seg: int, seg_len: int, n_facts: int,
                     key_pool, val_pool, qtok_id: int, pad_id: int, device,
                     generator=None, query_updated: bool = True):
    """-> x, y, m. seg0 states kj->vj once; seg1 restates ku->v'u; last seg
    queries a key. query_updated: target = v'u (latest). Else query a
    NON-updated fact (control: recency must not disturb ordinary recall)."""
    assert n_seg >= 3, "update task needs seg0 (facts), an update seg, a query seg"
    g = generator
    key_pool = torch.as_tensor(key_pool)
    val_pool = torch.as_tensor(val_pool)
    x = torch.full((batch, n_seg, seg_len), pad_id, dtype=torch.long)
    ans = torch.zeros(batch, n_seg, seg_len, dtype=torch.bool)

    for b in range(batch):
        perm_k = torch.randperm(len(key_pool), generator=g)
        perm_v = torch.randperm(len(val_pool), generator=g)
        ks = key_pool[perm_k[:n_facts]]
        vs = val_pool[perm_v[:n_facts]]
        pairs = []
        for k, v in zip(ks.tolist(), vs.tolist()):
            pairs += [k, v]

        # seg 0: prose with the facts stated ONCE at the front
        start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
        x[b, 0] = ids[start:start + seg_len]
        x[b, 0, :len(pairs)] = torch.tensor(pairs[:seg_len])

        # pick the fact to update and a NEW value distinct from every value used
        ju = int(torch.randint(n_facts, (1,), generator=g))
        v_new = int(val_pool[perm_v[n_facts]])       # next unused value token

        # seg 1: prose with the single update ku -> v'u at the front
        start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
        x[b, 1] = ids[start:start + seg_len]
        x[b, 1, :2] = torch.tensor([int(ks[ju]), v_new])

        # middle segments: pure prose distractors
        for s in range(2, n_seg - 1):
            start = int(torch.randint(len(ids) - seg_len, (1,), generator=g))
            x[b, s] = ids[start:start + seg_len]

        if query_updated:
            qk, target = int(ks[ju]), v_new
        else:
            jc = (ju + 1) % n_facts                   # a fact that was NOT updated
            qk, target = int(ks[jc]), int(vs[jc])
        tail = [qtok_id, qk, target][-seg_len:]
        x[b, n_seg - 1, seg_len - len(tail):] = torch.tensor(tail)
        ans[b, n_seg - 1, seg_len - 1] = True          # the (target) value token

    y = torch.empty_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    y[:, :, -1] = pad_id
    m = torch.zeros_like(ans)
    m[:, :, :-1] = ans[:, :, 1:]
    return x.to(device), y.to(device), m.to(device)
