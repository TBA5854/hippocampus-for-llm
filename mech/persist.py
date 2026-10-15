"""
persist.py — does a trained store survive the process, and where do its misses go?

Offline probes on a finished run's checkpoint (train.py --save); no training.

    # 1. write: facts + distractors go into the store, store is saved, process exits
    python -m mech.persist write    --run R.json --ckpt R.pt --dir D
    # 2. read: a FRESH process loads model + store and answers the query segment,
    #    which contains no facts — nothing but the store can supply them
    python -m mech.persist read     --run R.json --ckpt R.pt --dir D
    # where do matching misses land?
    python -m mech.persist taxonomy --run R.json --ckpt R.pt

Two processes on purpose. A same-process reload can pass because some state
(the centring mean, a cached tensor) never actually left memory; exiting
between write and read is what makes "the store persisted" a claim about the
file rather than about the interpreter.
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from mech.data import load
from mech.tasks import N_SYMBOLS, TASK_CHARS, _sym, mqar_vocab
from mech.train import build, get_batch


def restore(run: Path, ckpt: Path, batch: int, device):
    args = Namespace(**json.loads(run.read_text())["args"])
    args.batch = batch
    args.shuffle_store = False           # probes always read the model's own store
    if getattr(args, "task", None) in ("bpe_recall", "bpe_update", "bpe_hop",
                                       "bpe_chain", "bpe_collision"):
        return _restore_bpe(args, ckpt, device)
    tok, _, val_ids = load(args.data, extra_chars=TASK_CHARS)
    nsym = getattr(args, "n_symbols", N_SYMBOLS)   # old result JSONs predate the flag
    model = build(args, mqar_vocab(tok.vocab_size, nsym), device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return args, tok, val_ids, model


def _restore_bpe(args, ckpt: Path, device):
    """The bpe_recall analogue: real-word single-token keys over a BPE vocab.

    Rebuilds exactly what train.main() sets up for the task — the disjoint
    key/value word pools and query marker are a deterministic function of the
    tokenizer and n_facts, so a fresh process reconstructs the same facts, which
    is what makes the two-process reload a fair test."""
    from mech.bpe import BPETokenizer, load_books_bpe
    tok = BPETokenizer.load(args.bpe)
    _, val_ids = load_books_bpe(args.data_books, tok,
                                max_books=getattr(args, "max_books", None))
    words = tok.single_token_words()
    half = len(words) // 2
    args._key_pool = words[:half]
    args._val_pool = words[half:2 * half]
    args._qtok = words[-1]
    args._pad = tok.pad_id
    model = build(args, tok.vocab_size, device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return args, tok, val_ids, model


def answer_hits(logits, y, m):
    sel = m.reshape(-1)
    return int((logits.reshape(-1, logits.size(-1))[sel].argmax(-1)
                == y.reshape(-1)[sel]).sum()), int(sel.sum())


def _vocab_for(args, tok):
    if getattr(args, "task", None) == "bpe_recall":
        return tok.vocab_size
    return mqar_vocab(tok.vocab_size, getattr(args, "n_symbols", N_SYMBOLS))


@torch.no_grad()
def context(a, device):
    """How many context-windows does the store bridge?

    The facts are stated in segment 0 (one seg_len window); the query is a
    SEPARATE window (segments do not cross-attend — the store is the only
    channel). So (n_seg - 1) counts the windows of unrelated prose between the
    facts and the query that the store must reach across, and recall from the
    query alone measures whether it does. Sweep n_seg (the bridge distance) and
    memory_size M (the store's raw capacity): the bridge holds while the store
    has room for seg 0's facts plus the intervening prose (~n_seg x (seg_len-1)
    slots), and collapses once later prose evicts the facts. That collapse point
    is set by M, which is exactly how you 'store more': grow M, bridge farther.
    """
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    rows = []
    for M in a.mem_list:
        args.memory_size = M
        model = build(args, vocab, device)
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        model.eval()
        if getattr(a, "store_grow", False):
            _set_grow(model, True)                     # r100: append, don't lap
        for n_seg in a.nseg_list:
            args.n_seg = n_seg
            gen = torch.Generator().manual_seed(1234)
            hits = tot = 0
            for _ in range(a.iters):
                x, y, m = get_batch(val_ids, tok, args, device, gen)
                model.memory.reset(x.shape[0], device=device,
                                   dtype=model.tok_emb.weight.dtype)
                for s in range(n_seg - 1):            # facts (seg 0) + prose
                    _, h_write = model(x[:, s])
                    model.memory.commit(
                        h_write, score=getattr(model, "write_score", None))
                logits, _ = model(x[:, -1])           # query window ALONE
                h, t = answer_hits(logits, y[:, -1], m[:, -1])
                hits += h; tot += t
            slots = int(model.memory.store.n_stored[0])
            row = {"M": M, "n_seg": n_seg, "bridge_windows": n_seg - 1,
                   "written_tokens": (n_seg - 1) * args.seg_len,
                   "slots_used": slots, "slots_capacity": M,
                   "recall": hits / tot, "n": tot}
            rows.append(row)
            print(f"M={M:5d}  n_seg={n_seg:3d}  bridge={n_seg-1:3d}win "
                  f"({(n_seg-1)*args.seg_len:5d} tok)  slots={slots}/{M}  "
                  f"recall={hits/tot:6.1%}", flush=True)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "seg_len": args.seg_len,
                                     "rows": rows}, indent=2))


def _set_readout(model, readout, k):
    """Flip the (frozen) MemConfig's readout at eval time, on every cfg the read
    path touches. `topk` blends only the k highest-scoring slots — the model then
    answers from at most k slots per query regardless of how large the store is."""
    cfgs = {id(model.memory.cfg): model.memory.cfg}
    for h in model.memory.read_heads.values():
        cfgs[id(h.cfg)] = h.cfg
    for c in cfgs.values():
        object.__setattr__(c, "readout", readout)
        if k:
            object.__setattr__(c, "top_k", int(k))


@torch.no_grad()
def sparse(a, device):
    """The north-star half: a store far larger than the context window, answered
    by touching only a few slots per query.

    Fill a big store that bridges many context windows (like `context`), then read
    the query with a top-k readout for shrinking k. If recall at k=1 matches the
    dense (graded, all-M) read, the store holds K× the window AND each query needs
    only ONE of its M slots — 'more context than the window, query only what's
    required', measured together.
    """
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    args.memory_size = a.mem_list[0]
    args.n_seg = a.nseg_list[0]
    model = build(args, vocab, device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    st = model.memory.store
    orig = model.memory.cfg.readout
    if getattr(a, "rare_keys", False):                 # r102: low-collision fact keys
        rare = tok.single_token_words()[-max(4 * args.n_facts, 500):]
        args._key_pool = rare                          # rarest words: seldom recur in prose
    if getattr(a, "store_grow", False):
        _set_grow(model, True)                         # r100: grow + sparse read together
    if getattr(a, "shuffle", False):
        # matched control: every read pulls a neighbour example's store, so a
        # correct answer cannot come from the store's own contents -> chance.
        for c in ({id(model.memory.cfg): model.memory.cfg} |
                  {id(h.cfg): h.cfg for h in model.memory.read_heads.values()}).values():
            object.__setattr__(c, "shuffle_store", True)
    rows = []
    # topk_list entries: a positive k => top-k read; 0 => the dense graded baseline
    for k in a.topk_list:
        if k == 0:
            _set_readout(model, "graded", None)
        else:
            _set_readout(model, "topk", k)
        gen = torch.Generator().manual_seed(1234)
        hits = tot = 0
        total = 0
        for _ in range(a.iters):
            x, y, m = get_batch(val_ids, tok, args, device, gen)
            model.memory.reset(x.shape[0], device=device,
                               dtype=model.tok_emb.weight.dtype)
            for s in range(args.n_seg - 1):
                _, h_write = model(x[:, s])
                model.memory.commit(h_write, score=getattr(model, "write_score", None))
            total = int(st.n_stored[0])
            logits, _ = model(x[:, -1])
            h, t = answer_hits(logits, y[:, -1], m[:, -1])
            hits += h; tot += t
        label = "graded(all)" if k == 0 else f"top-{k}"
        touched = total if k == 0 else min(k, total)
        rows.append({"read": label, "k": k, "slots_in_store": total,
                     "slots_touched": touched, "bridge_windows": args.n_seg - 1,
                     "recall": hits / tot, "n": tot})
        print(f"{label:12s}  store={total:5d} slots ({args.n_seg-1} windows)  "
              f"touched={touched:5d}/{total}  recall={hits/tot:6.1%}", flush=True)
    _set_readout(model, orig, None)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "seg_len": args.seg_len,
                                     "memory_size": args.memory_size, "rows": rows}, indent=2))


def passkey(a, device):
    """Passkey retrieval (needle-in-a-haystack), the standard long-context benchmark.
    A fact (the passkey) is planted in ONE of many filler windows and queried at the
    end; we sweep the needle's DEPTH (which window holds it) at a fixed context length.
    An attention model suffers 'lost in the middle' — recall dips for mid-context
    needles. Our store is content-addressed by a position-free layer-0 key, so we
    predict recall is INVARIANT to depth. Top-1 read; --shuffle is the matched
    scrambled-store control (-> chance)."""
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    args.memory_size = a.mem_list[0]
    args.n_seg = a.nseg_list[0]
    model = build(args, vocab, device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    st = model.memory.store
    if getattr(a, "rare_keys", False):
        rare = tok.single_token_words()[-max(4 * args.n_facts, 500):]
        args._key_pool = rare
    if getattr(a, "store_grow", False):
        _set_grow(model, True)
    if getattr(a, "shuffle", False):
        for c in ({id(model.memory.cfg): model.memory.cfg} |
                  {id(h.cfg): h.cfg for h in model.memory.read_heads.values()}).values():
            object.__setattr__(c, "shuffle_store", True)
    _set_readout(model, "topk", 1)
    ctx = args.n_seg - 1                                # number of context windows
    fracs = a.depth_list                               # needle depth as a fraction of ctx
    rows = []
    for f in fracs:
        d = min(max(int(round(f * (ctx - 1))), 0), ctx - 1)
        args.needle_seg = d
        gen = torch.Generator().manual_seed(1234)
        hits = tot = total = 0
        for _ in range(a.iters):
            x, y, m = get_batch(val_ids, tok, args, device, gen)
            model.memory.reset(x.shape[0], device=device,
                               dtype=model.tok_emb.weight.dtype)
            for s in range(args.n_seg - 1):
                _, h_write = model(x[:, s])
                model.memory.commit(h_write, score=getattr(model, "write_score", None))
            total = int(st.n_stored[0])
            logits, _ = model(x[:, -1])
            h, t = answer_hits(logits, y[:, -1], m[:, -1])
            hits += h; tot += t
        rows.append({"depth_frac": f, "needle_seg": d, "ctx_windows": ctx,
                     "tokens": ctx * args.seg_len, "slots_in_store": total,
                     "recall": hits / tot, "n": tot})
        print(f"depth={f:4.2f} (window {d:5d}/{ctx})  store={total:6d} slots  "
              f"top-1 recall={hits/tot:6.1%}", flush=True)
    _set_readout(model, model.memory.cfg.readout, None)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "seg_len": args.seg_len,
                                     "memory_size": args.memory_size, "ctx_windows": ctx,
                                     "shuffle": bool(getattr(a, "shuffle", False)),
                                     "rare_keys": bool(getattr(a, "rare_keys", False)),
                                     "rows": rows}, indent=2))


def _set_grow(model, on):
    """Flip store_grow on every cfg the store/read path sees (r100 eval-time toggle)."""
    cfgs = {id(model.memory.cfg): model.memory.cfg}
    for h in model.memory.read_heads.values():
        cfgs[id(h.cfg)] = h.cfg
    for c in cfgs.values():
        object.__setattr__(c, "store_grow", bool(on))


def _set_pool(model, mode, P, anchor=None):
    cfgs = {id(model.memory.cfg): model.memory.cfg}
    for h in model.memory.read_heads.values():
        cfgs[id(h.cfg)] = h.cfg
    for c in cfgs.values():
        object.__setattr__(c, "read_pool", mode)
        object.__setattr__(c, "pool_size", int(P))
        object.__setattr__(c, "pool_anchor_layer", anchor)


@torch.no_grad()
def pool(a, device):
    """CSA2 candidate-pool read (r64): restrict the read to index-selected slots.

    Fill a big store, then read the query under each pool mode — `none` (dense,
    all M), `token` (slots whose stored token matches the query's), `ann` (slots
    in the query's LSH bucket) — and report recall plus the mean pool size (the
    candidates a real index would score, vs M for dense). If token/ann recall
    matches dense with pool <<< M, the read is O(pool), not O(M).
    """
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    args.memory_size = a.mem_list[0]
    args.n_seg = a.nseg_list[0]
    model = build(args, vocab, device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    st = model.memory.store
    mem = model.memory
    anchor = a.pool_anchor if getattr(a, "pool_anchor", None) is not None and a.pool_anchor >= 0 else None
    rows = []
    for mode in a.pool_modes:
        _set_pool(model, mode, a.pool_size, anchor)
        gen = torch.Generator().manual_seed(1234)
        hits = tot = 0
        psz = []
        total = 0
        for _ in range(a.iters):
            x, y, m = get_batch(val_ids, tok, args, device, gen)
            B = x.shape[0]
            model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
            for s in range(args.n_seg - 1):
                _, h_write = model(x[:, s])
                model.memory.commit(h_write, score=getattr(model, "write_score", None))
            total = int(st.n_stored[0])
            logits, _ = model(x[:, -1])               # read uses the pool
            if mode != "none":                        # measure candidates at the answer
                pm = mem.build_pool(st, model.tok_emb(x[:, -1]))     # [B, S, M]
                am = m[:, -1]                                        # answer mask [B, S]
                sz = pm.sum(-1).float()                              # [B, S]
                psz.append(float((sz * am).sum() / am.sum().clamp_min(1)))
            h, t = answer_hits(logits, y[:, -1], m[:, -1])
            hits += h; tot += t
        cand = (sum(psz) / len(psz)) if psz else float(total)
        rows.append({"pool": mode, "store": total, "candidates": cand,
                     "bridge_windows": args.n_seg - 1, "recall": hits / tot, "n": tot})
        print(f"{mode:6s}  store={total:5d}  candidates={cand:8.1f}/{total}  "
              f"recall={hits/tot:6.1%}", flush=True)
    _set_pool(model, "none", a.pool_size)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "pool_size": a.pool_size,
                                     "pool_anchor": anchor, "memory_size": args.memory_size,
                                     "rows": rows}, indent=2))


def _fakequant(x, bits):
    """Symmetric int-N fake-quant with a per-slot scale (round-trip through N bits).
    Approximates CSA2's FP4 KV cache: the question is how few bits a stored code
    survives, i.e. how much smaller the store can be per slot."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    return torch.round(x / scale).clamp(-qmax, qmax) * scale


@torch.no_grad()
def quant(a, device):
    """CSA2 axis 3 (r65): quantize the stored codes to N bits and read.

    Fill a big store, fake-quantize keys and/or values to each bit width, read the
    query, report recall + bytes/slot. With the token pool, matching is by token id
    (immune to key quantization) so only the value bits matter — measured via
    --pool-modes. How few bits the store survives = how much smaller it can be.
    """
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    args.memory_size = a.mem_list[0]
    args.n_seg = a.nseg_list[0]
    model = build(args, vocab, device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    st = model.memory.store
    what = a.quant_what
    rows = []
    for mode in a.pool_modes:
        _set_pool(model, mode, a.pool_size)
        for bits in a.quant_bits:
            gen = torch.Generator().manual_seed(1234)
            hits = tot = 0
            total = 0
            for _ in range(a.iters):
                x, y, m = get_batch(val_ids, tok, args, device, gen)
                model.memory.reset(x.shape[0], device=device,
                                   dtype=model.tok_emb.weight.dtype)
                for s in range(args.n_seg - 1):
                    _, h_write = model(x[:, s])
                    model.memory.commit(h_write, score=getattr(model, "write_score", None))
                total = int(st.n_stored[0])
                if what in ("keys", "both"):
                    st.keys = _fakequant(st.keys, bits)
                if what in ("values", "both"):
                    st.values = _fakequant(st.values, bits)
                logits, _ = model(x[:, -1])
                h, t = answer_hits(logits, y[:, -1], m[:, -1])
                hits += h; tot += t
            bytes_slot = (bits * args.d_mem / 8) * (2 if what == "both" else 1)
            rows.append({"pool": mode, "bits": bits, "quant_what": what,
                         "bytes_per_slot": bytes_slot, "store": total,
                         "recall": hits / tot, "n": tot})
            print(f"{mode:6s} {bits:2d}bit ({what:6s})  {bytes_slot:6.0f} B/slot  "
                  f"recall={hits/tot:6.1%}", flush=True)
    _set_pool(model, "none", a.pool_size)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "quant_what": what,
                                     "memory_size": args.memory_size, "rows": rows}, indent=2))


def _keepB(policy, total, M_phys, F, B, score, device):
    """[B, kept] slot indices a physical store of size M_phys keeps.

    ring       — the most recent M_phys (r51: seg 0's facts, written first, are
                 evicted first). Same slots for every example.
    oracle     — protect the F fact slots [0, F) and fill with the most recent
                 prose. Upper bound: never drops a fact. Same for every example.
    surprisal  — keep the M_phys highest-surprisal slots, PER example, from the
                 write-time -log p(token). No labels: a random-word fact token is
                 unpredictable, prose is not, so this is the deployable guess at
                 which slots are facts. `score` is [B, total].
    """
    if M_phys >= total:
        return torch.arange(total, device=device)[None].expand(B, total)
    if policy == "ring":
        idx = torch.arange(total - M_phys, total, device=device)
        return idx[None].expand(B, M_phys)
    if policy == "oracle":
        rem = max(M_phys - F, 0)
        facts = torch.arange(0, min(F, M_phys), device=device)
        prose = (torch.arange(total - rem, total, device=device) if rem else
                 torch.empty(0, dtype=torch.long, device=device))
        idx = torch.cat([facts, prose])
        return idx[None].expand(B, idx.numel())
    if policy == "surprisal":
        return score[:, :total].topk(M_phys, dim=1).indices.sort(dim=1).values
    raise ValueError(policy)


def _surprisal(logits, toks):
    """[B, seg_len-1] per-position -log p(token) under the model's own head."""
    lp = torch.log_softmax(logits[:, :-1], dim=-1)
    return -lp.gather(-1, toks[:, 1:, None]).squeeze(-1)


@torch.no_grad()
def evict(a, device):
    """Does protecting the fact slots turn r51's hard cliff into graceful reach,
    and can an UNSUPERVISED score (surprisal) do it without being told which
    slots are facts?

    r51 showed a plain ring buffer caps effective context at ~M/128 windows and
    collapses to 0% the moment prose laps segment 0's facts. Here we write every
    segment to a large store (no eviction), then PRUNE to a fixed physical size M
    under each policy and read the query from the pruned store. `ring` reproduces
    r51; `oracle` keeps the F=seg_len-value_shift fact slots (upper bound);
    `surprisal` keeps the highest -log p(token) slots per example (r53: the
    deployable approximation — facts are random words, so high-surprisal). If
    surprisal tracks oracle, "keep the right slots" needs no labels.
    """
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    vocab = _vocab_for(args, tok)
    F = args.seg_len - args.value_shift                 # fact-segment slot count
    S = F                                               # slots committed per seg
    big = max(a.nseg_list) * args.seg_len + 64          # never overflows
    args.memory_size = big
    model = build(args, vocab, device)
    model.load_state_dict(torch.load(a.ckpt, map_location=device))
    model.eval()
    st = model.memory.store
    d = st.keys.shape[-1]
    rows = []
    for n_seg in a.nseg_list:
        args.n_seg = n_seg
        for M_phys in a.mem_list:
            for policy in a.policies:
                gen = torch.Generator().manual_seed(1234)
                hits = tot = 0
                for _ in range(a.iters):
                    x, y, m = get_batch(val_ids, tok, args, device, gen)
                    B = x.shape[0]
                    model.memory.reset(B, device=device,
                                       dtype=model.tok_emb.weight.dtype)
                    surp = policy in ("surprisal", "surprisal_dedup")
                    score = torch.full((B, big), float("-inf"), device=device)
                    token = torch.zeros((B, big), dtype=torch.long, device=device)
                    for s in range(n_seg - 1):          # facts (seg 0) + prose
                        logits, h_write = model(x[:, s])
                        model.memory.commit(
                            h_write, score=getattr(model, "write_score", None))
                        if surp:
                            score[:, s * S:(s + 1) * S] = _surprisal(logits, x[:, s])
                            token[:, s * S:(s + 1) * S] = x[:, s, 1:1 + S]
                    total = int(st.n_stored[0])
                    pol = policy
                    if policy == "surprisal_dedup":
                        # r55: among slots holding the SAME token, keep only the
                        # highest-surprisal copy — so a tiled fact's low-surprisal
                        # repeats stop crowding the ranking (r54's failure mode).
                        sc, tk = score[:, :total].clone(), token[:, :total]
                        mx = torch.full((B, vocab), float("-inf"), device=device)
                        mx.scatter_reduce_(1, tk, sc, reduce="amax", include_self=True)
                        keeper = sc >= mx.gather(1, tk)
                        score[:, :total] = sc.masked_fill(~keeper, float("-inf"))
                        pol = "surprisal"               # then top-M by deduped score
                    keep = _keepB(pol, total, M_phys, F, B, score, device)
                    kept = keep.shape[1]
                    g = keep[:, :, None].expand(B, kept, d)
                    st.keys[:, :kept] = torch.gather(st.keys, 1, g)
                    st.values[:, :kept] = torch.gather(st.values, 1, g)
                    st.n_stored[:] = kept
                    st.ptr[:] = kept % big
                    logits, _ = model(x[:, -1])         # read from pruned store
                    h, t = answer_hits(logits, y[:, -1], m[:, -1])
                    hits += h; tot += t
                row = {"policy": policy, "M_phys": M_phys, "n_seg": n_seg,
                       "bridge_windows": n_seg - 1, "kept": kept,
                       "recall": hits / tot, "n": tot}
                rows.append(row)
                print(f"{policy:9s}  M={M_phys:5d}  n_seg={n_seg:3d}  "
                      f"bridge={n_seg-1:3d}win  kept={kept}/{M_phys}  "
                      f"recall={hits/tot:6.1%}", flush=True)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "seg_len": args.seg_len,
                                     "F": F, "rows": rows}, indent=2))


def _snapshot(store):
    """Clone every buffer of a store (keys/values/n_stored/ptr/seq/tok/... )."""
    return {n: b.clone() for n, b in store.named_buffers()}


def _restore_snap(store, snap):
    for n, b in store.named_buffers():
        if n in snap and b.shape == snap[n].shape:
            b.copy_(snap[n])


@torch.no_grad()
def _fill(model, x, device):
    """Commit the fact/prose segments (all but the last) into the model's store and
    return a snapshot of the filled store."""
    model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
    for s in range(x.shape[1] - 1):
        _, h = model(x[:, s])
        model.memory.commit(h, score=getattr(model, "write_score", None))
    return _snapshot(model.memory.store)


@torch.no_grad()
def _answer(model, snap, xq, y, m, device, shuffle=False):
    """Load a store snapshot into `model` and answer the query window ALONE."""
    model.memory.reset(xq.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
    _restore_snap(model.memory.store, snap)
    if shuffle:                                   # scrambled-store control -> chance
        st = model.memory.store
        perm = torch.randperm(st.keys.shape[0], device=device)
        for n, b in st.named_buffers():
            if b.shape[0] == perm.shape[0]:
                b.copy_(b[perm])
    logits, _ = model(xq)
    return answer_hits(logits, y, m)


@torch.no_grad()
def swap(a, device):
    """Memory transplant: weights are stateless, the store is the state (r86).

    Same frozen model, two independent stores (two disjoint fact sets). If the
    store is the state, then which memories the model has is decided entirely by
    which store is loaded: model A answers query1 from store1 and query2 from
    store2, but answers the WRONG store at chance (swapping the store swaps the
    memory). If a --run-b/--ckpt-b foreign model is given, we force its (matching-
    shape) store into A and vice-versa, bypassing the save/load fingerprint that
    normally refuses a foreign store: a store written by a DIFFERENT model should
    NOT read back, because its keys are that model's token embeddings and the
    exact-match query is this model's -- memory is portable across store-swaps on
    the SAME weights, but bound to the weights that wrote it.
    """
    argsA, tokA, valA, modelA = restore(a.run, a.ckpt, a.batch, device)
    gen = torch.Generator().manual_seed(1234)
    x1, y1, m1 = get_batch(valA, tokA, argsA, device, gen)   # fact set 1
    x2, y2, m2 = get_batch(valA, tokA, argsA, device, gen)   # fact set 2 (gen advanced)
    snap1 = _fill(modelA, x1, device)
    snap2 = _fill(modelA, x2, device)

    def rec(model, snap, xq, y, m, shuffle=False):
        h, t = _answer(model, snap, xq[:, -1], y[:, -1], m[:, -1], device, shuffle)
        return h / t

    rows = {
        "A_store1_query1_own":  rec(modelA, snap1, x1, y1, m1),
        "A_store2_query2_own":  rec(modelA, snap2, x2, y2, m2),
        "A_store2_query1_swap": rec(modelA, snap2, x1, y1, m1),   # wrong store -> chance
        "A_store1_query2_swap": rec(modelA, snap1, x2, y2, m2),
        "A_store1_query1_scrambled": rec(modelA, snap1, x1, y1, m1, shuffle=True),
    }
    if getattr(a, "run_b", None):
        argsB, tokB, valB, modelB = restore(a.run_b, a.ckpt_b, a.batch, device)
        gb = torch.Generator().manual_seed(4321)
        xb, yb, mb = get_batch(valB, tokB, argsB, device, gb)
        snapB = _fill(modelB, xb, device)
        rows["B_storeB_queryB_own"] = rec(modelB, snapB, xb, yb, mb)
        sa = modelA.memory.store.keys.shape[1:]
        sb = modelB.memory.store.keys.shape[1:]
        if sa == sb:
            # FOREIGN transplant: B reads A's store on A's own facts (same store +
            # query as A_store1_query1_own -> isolates the model as the only change).
            rows["B_store1_query1_foreign"] = rec(modelB, snap1, x1, y1, m1)
            rows["A_storeB_queryB_foreign"] = rec(modelA, snapB, xb, yb, mb)
        else:
            rows["foreign_note"] = f"shape mismatch A{tuple(sa)} vs B{tuple(sb)} (d_mem differs)"
    print(json.dumps(rows, indent=2))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2))


@torch.no_grad()
def write(a, device):
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    a.dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(1234)
    fp = model.fingerprint()
    hits = tot = 0
    for i in range(a.iters):
        x, y, m = get_batch(val_ids, tok, args, device, gen)
        n_seg = x.shape[1]
        model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
        for s in range(n_seg - 1):                        # facts + distractors only
            _, h_write = model(x[:, s])
            model.memory.commit(h_write, score=getattr(model, "write_score", None))
        model.memory.store.save(a.dir / f"store_{i}.pt", fp)
        torch.save({"x": x[:, -1].cpu(), "y": y[:, -1].cpu(), "m": m[:, -1].cpu()},
                   a.dir / f"query_{i}.pt")
        logits, _ = model(x[:, -1])                       # in-process answer, for reference
        h, t = answer_hits(logits, y[:, -1], m[:, -1]); hits += h; tot += t
    size = sum(p.stat().st_size for p in a.dir.glob("store_*.pt"))
    out = {"in_process_recall": hits / tot, "n": tot, "fingerprint": fp,
           "store_bytes_per_file": size // a.iters,
           "slots_per_example": int(model.memory.store.n_stored[0])}
    (a.dir / "write.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


@torch.no_grad()
def read(a, device):
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    fp = model.fingerprint()
    hits = tot = 0
    for i in range(a.iters):
        q = torch.load(a.dir / f"query_{i}.pt")
        x, y, m = q["x"].to(device), q["y"].to(device), q["m"].to(device)
        model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
        # --mismatch pairs each query with ANOTHER batch's store file: the
        # persistence analogue of the scrambled-store control. If answers come
        # from the file's contents, this must fall to chance.
        j = (i + 1) % a.iters if a.mismatch else i
        model.memory.store.load(a.dir / f"store_{j}.pt", fp, device)
        logits, _ = model(x)                              # the query segment ALONE
        h, t = answer_hits(logits, y, m); hits += h; tot += t
    ref = json.loads((a.dir / "write.json").read_text())
    key = "mismatched_store_recall" if a.mismatch else "reloaded_recall"
    out = {key: hits / tot, "in_process_recall": ref["in_process_recall"], "n": tot}
    (a.dir / ("read_mismatch.json" if a.mismatch else "read.json")).write_text(
        json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


@torch.no_grad()
def taxonomy(a, device):
    """Classify the slot each query's read lands on by what its key position
    holds. Tells a matching miss that hit ANOTHER FACT (key interference, which
    worsens with fact count) from one that hit a value slot or distractor text
    (which worsens with store size)."""
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    head = next(iter(model.memory.read_heads.values()))
    nsym = getattr(args, "n_symbols", N_SYMBOLS)
    kbase, vbase, qtok = _sym(tok.vocab_size, nsym)
    gen = torch.Generator().manual_seed(1234)
    shift = model.memory.cfg.value_shift
    cats = dict.fromkeys(["correct", "other_fact_key", "value_symbol", "query_marker",
                          "distractor_text", "fact_segment_other"], 0)
    by_answer = {True: dict(cats), False: dict(cats)}
    n = 0
    for _ in range(a.iters):
        x, y, m = get_batch(val_ids, tok, args, device, gen)
        B, n_seg, S = x.shape
        model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
        for s in range(n_seg - 1):
            _, h_write = model(x[:, s])
            model.memory.commit(h_write, score=getattr(model, "write_score", None))
        head.stash = {}
        logits, _ = model(x[:, -1])
        w = head.stash["w"]
        head.stash = None
        bidx = torch.arange(B, device=device)
        pos = m[:, -1].float().argmax(dim=1)
        qkey = x[bidx, -1, pos]
        slot = w[bidx, pos].argmax(dim=1)
        per = S - shift
        seg, p = slot // per, slot % per
        tokn = x[bidx, seg, p]
        right = logits[bidx, pos].argmax(-1) == y[bidx, -1, pos]
        for b in range(B):
            t, sg = int(tokn[b]), int(seg[b])
            if sg == 0 and t == int(qkey[b]):
                c = "correct"
            elif kbase <= t < kbase + nsym:
                c = "other_fact_key" if sg == 0 else "distractor_text"
            elif vbase <= t < vbase + nsym:
                c = "value_symbol"
            elif t == qtok:
                c = "query_marker"
            else:
                c = "distractor_text" if sg > 0 else "fact_segment_other"
            by_answer[bool(right[b])][c] += 1
            n += 1
    out = {"n": n,
           "answer_right": {k: v / n for k, v in by_answer[True].items()},
           "answer_wrong": {k: v / n for k, v in by_answer[False].items()}}
    print(json.dumps(out, indent=2))
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))


_SYN_PAIRS = [
    ("big", "large"), ("small", "little"), ("fast", "quick"), ("happy", "glad"),
    ("begin", "start"), ("buy", "purchase"), ("close", "shut"), ("cold", "chilly"),
    ("angry", "furious"), ("smart", "clever"), ("rich", "wealthy"), ("sad", "unhappy"),
    ("hard", "difficult"), ("easy", "simple"), ("old", "ancient"), ("strong", "powerful"),
    ("weak", "feeble"), ("beautiful", "lovely"), ("famous", "renowned"), ("quiet", "silent"),
    ("loud", "noisy"), ("brave", "bold"), ("afraid", "scared"), ("tired", "weary"),
    ("wet", "damp"), ("clean", "spotless"), ("dirty", "filthy"), ("empty", "vacant"),
    ("whole", "entire"), ("keep", "retain"), ("show", "reveal"), ("hide", "conceal"),
    ("help", "assist"), ("break", "shatter"), ("gaze", "stare"), ("talk", "speak"),
    ("walk", "stroll"), ("shout", "yell"), ("jump", "leap"), ("laugh", "giggle"),
    ("weep", "sob"), ("story", "tale"), ("answer", "reply"), ("mistake", "error"),
    ("idea", "notion"), ("anger", "rage"), ("wise", "sage"), ("odd", "strange"),
]


@torch.no_grad()
def semantic(a, device):
    """Is the emergent fuzzy recall SEMANTIC, not just embedding-geometry? Plant facts keyed
    by a word, then query with a curated SYNONYM (single-token) vs the exact word vs a random
    token. If synonyms retrieve well above the random control, the store's approximate recall
    tracks meaning (to the extent the token embeddings encode it), not surface form."""
    import torch.nn.functional as F
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    stoi = tok.stoi
    wid = lambda w: stoi.get("Ġ" + w)                 # 'Ġword' = word-initial token
    valid = [(wid(x), wid(z)) for x, z in _SYN_PAIRS if wid(x) is not None and wid(z) is not None]
    n = len(valid)
    if n < 4:
        print(f"only {n} single-token synonym pairs in this vocab; abort"); return
    w1 = [p[0] for p in valid]; w2 = [p[1] for p in valid]
    E = F.normalize(model.tok_emb.weight, dim=-1)
    pad, qtok, S = args._pad, args._qtok, args.seg_len
    gen = torch.Generator().manual_seed(1234)
    exclude = set(w1) | set(w2)
    words = [i for i in tok.single_token_words() if i not in exclude]
    vals = torch.tensor(words)[torch.randperm(len(words), generator=gen)[:n]].tolist()
    rnd = torch.tensor(words)[torch.randperm(len(words), generator=gen)[:n]].tolist()  # random-key control
    facts = []
    for i in range(n):
        facts += [w1[i], vals[i]]
    reps = S // max(len(facts), 1) + 1
    seg0 = torch.tensor((facts * reps)[:S])
    B = n
    x = torch.full((B, args.n_seg, S), pad, dtype=torch.long)
    for b in range(B):
        x[b, 0] = seg0
        for s in range(1, args.n_seg - 1):
            st = int(torch.randint(len(val_ids) - S, (1,), generator=gen))
            x[b, s] = val_ids[st:st + S]
    x = x.to(device)
    model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
    for s in range(args.n_seg - 1):                        # fill the store once
        _, hw = model(x[:, s]); model.memory.commit(hw, score=getattr(model, "write_score", None))
    store_snap = _snapshot(model.memory.store)
    vt = torch.tensor(vals, device=device)

    def probe(keys):
        _restore_snap(model.memory.store, store_snap)      # same filled store each condition
        q = torch.full((B, S), pad, dtype=torch.long, device=device)
        q[:, S - 3] = qtok                                 # match training: [qtok, key, .]
        q[:, S - 2] = torch.tensor(keys, device=device)    # key at seg_len-2
        logits, _ = model(q)
        pred = logits[:, S - 2].argmax(-1)                 # key position predicts the value
        return float((pred == vt).float().mean())

    cos_syn = float(F.cosine_similarity(E[torch.tensor(w1)], E[torch.tensor(w2)]).mean())
    cos_rnd = float(F.cosine_similarity(E[torch.tensor(w1)], E[torch.tensor(rnd)]).mean())
    r_exact = probe(w1); r_syn = probe(w2); r_rand = probe(rnd)
    print(f"semantic probe (n={n} single-token synonym pairs):", flush=True)
    print(f"  exact (query=word)     recall {r_exact:6.1%}", flush=True)
    print(f"  synonym (query=synonym) recall {r_syn:6.1%}   [cos(word,syn)={cos_syn:.3f}]", flush=True)
    print(f"  random (query=random)   recall {r_rand:6.1%}   [cos(word,rnd)={cos_rnd:.3f}]  chance~{1/n:.1%}", flush=True)
    if a.out:
        a.out.write_text(json.dumps({"n_pairs": n, "recall_exact": r_exact, "recall_synonym": r_syn,
                                     "recall_random": r_rand, "cos_synonym": cos_syn,
                                     "cos_random": cos_rnd}, indent=2))


@torch.no_grad()
def fuzzy(a, device):
    """Is fuzzy/semantic recall ALREADY there? The read is argmax-cosine over stored keys,
    so a query token that is a NEAR-EMBEDDING NEIGHBOR of a stored key (a token the model
    was never told about) may still land on that key's slot and retrieve its value. We fill
    the store with exact-key facts, then replace each query key with its k-th nearest
    embedding neighbor (excluding any token that appears in the fact segment, so the only
    nearby stored slot is the intended one) and measure recall. rank 0 = exact baseline;
    if recall stays high at rank 1/2/..., the exact match is really a nearest match and
    fuzzy recall is emergent."""
    import torch.nn.functional as F
    from mech.tasks import bpe_recall_batch
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    E = F.normalize(model.tok_emb.weight, dim=-1)              # [V, d] unit rows
    sb = getattr(a, "settle_beta", 0.0)
    if sb and sb > 0:                                          # (b) override read sharpness
        cfgs = {id(model.memory.cfg): model.memory.cfg}
        for h in model.memory.read_heads.values():
            cfgs[id(h.cfg)] = h.cfg
        for c in cfgs.values():
            object.__setattr__(c, "beta", float(sb))
            object.__setattr__(c, "beta_max", float(sb))
        print(f"[settle beta overridden to {sb}]", flush=True)
    gen = torch.Generator().manual_seed(1234)
    rows = []
    for rank in a.nn_ranks:
        hits = tot = 0
        cos_sum = cos_n = 0.0
        for _ in range(a.iters):
            x, y, m = bpe_recall_batch(val_ids, a.batch, args.n_seg, args.seg_len,
                                       args.n_facts, args._key_pool, args._val_pool,
                                       args._qtok, args._pad, device, gen)
            B = x.shape[0]
            model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
            for s in range(args.n_seg - 1):
                _, hw = model(x[:, s])
                model.memory.commit(hw, score=getattr(model, "write_score", None))
            seg = x[:, -1].clone()
            bidx = torch.arange(B, device=device)
            qpos = m[:, -1].float().argmax(1)                 # the query-key position
            if rank > 0:
                ktok = seg[bidx, qpos]                        # [B] the true key tokens
                sims = E[ktok] @ E.T                          # [B, V] cosine to every token
                for b in range(B):                            # exclude the fact tokens + markers
                    sims[b, x[b, 0].unique()] = -1e9
                    sims[b, args._qtok] = -1e9
                    if args._pad is not None and args._pad >= 0:
                        sims[b, args._pad] = -1e9
                nn = sims.topk(rank, dim=1).indices[:, rank - 1]   # k-th nearest non-fact token
                cos_sum += float(sims.gather(1, nn[:, None]).mean()); cos_n += 1
                seg[bidx, qpos] = nn                          # query with the neighbour
            logits, _ = model(seg)
            h, t = answer_hits(logits, y[:, -1], m[:, -1]); hits += h; tot += t
        cos = (cos_sum / cos_n) if cos_n else 1.0
        rows.append({"nn_rank": rank, "neighbor_cos": cos, "recall": hits / tot, "n": tot})
        print(f"nn_rank={rank}  neighbor_cos={cos:.3f}  recall={hits/tot:6.1%}  n={tot}",
              flush=True)
    if a.out:
        a.out.write_text(json.dumps({"n_facts": args.n_facts, "rows": rows}, indent=2))


@torch.no_grad()
def chain_ar(a, device):
    """Autoregressive multi-hop eval: the HONEST CoT-over-memory number.

    The training/eval on bpe_chain is teacher-forced — each hop's block [q t_i t_{i+1}]
    contains the TRUE intermediate t_i. Here we feed the model its OWN emitted intermediate:
    generate t_{i+1} from block i, then place that emitted token as block i+1's query key,
    and measure whether the chain reaches the true tail through the model's own steps.
    """
    from mech.tasks import bpe_chain_batch
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    n_hops = a.n_hops if getattr(a, "n_hops", 0) else getattr(args, "n_hops", 2)
    qtok, pad, S = args._qtok, args._pad, args.seg_len
    off = S - 3 * n_hops
    gen = torch.Generator().manual_seed(1234)
    final_hits = tot = 0
    hop_hits = [0] * n_hops
    for _ in range(a.iters):
        x, y, m = bpe_chain_batch(val_ids, a.batch, args.n_seg, S, args.n_facts,
                                  args._key_pool, args._val_pool, qtok, pad, device, gen,
                                  n_hops=n_hops, broken=False,
                                  n_chains=getattr(args, "n_chains", 4))
        B = x.shape[0]
        seg = x[:, -1]
        t0 = seg[:, off + 1]                                   # chain head
        true_next = [seg[:, off + 3 * i + 2] for i in range(n_hops)]   # true t_{i+1}
        target = true_next[-1]
        model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
        for s in range(args.n_seg - 1):                        # fill the store
            _, hw = model(x[:, s])
            model.memory.commit(hw, score=getattr(model, "write_score", None))
        emitted = [t0]                                         # t0 is given
        for i in range(n_hops):
            seg_ar = torch.full((B, S), pad, dtype=torch.long, device=device)
            for j in range(i):                                 # completed blocks (own emissions)
                seg_ar[:, off + 3 * j] = qtok
                seg_ar[:, off + 3 * j + 1] = emitted[j]
                seg_ar[:, off + 3 * j + 2] = emitted[j + 1]
            seg_ar[:, off + 3 * i] = qtok                      # current block: [q emitted_i]
            seg_ar[:, off + 3 * i + 1] = emitted[i]
            logits, _ = model(seg_ar)                          # store already filled
            pred = logits[:, off + 3 * i + 1].argmax(-1)       # predicts t_{i+1}
            emitted.append(pred)
            hop_hits[i] += int((pred == true_next[i]).sum())
        final_hits += int((emitted[-1] == target).sum())
        tot += B
    per_hop = [h / tot for h in hop_hits]
    print(f"AR {n_hops}-hop: final(own intermediates) {final_hits/tot:.1%}  "
          f"per-hop {['%.1f%%'%(p*100) for p in per_hop]}  n={tot}", flush=True)
    if a.out:
        a.out.write_text(json.dumps({"n_hops": n_hops, "final_autoregressive": final_hits/tot,
                                     "per_hop": per_hop, "n": tot}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write", "read", "taxonomy", "context", "evict",
                                     "sparse", "pool", "quant", "swap", "chain_ar", "fuzzy",
                                     "semantic", "passkey"])
    ap.add_argument("--n-hops", type=int, default=0,
                    help="chain_ar: override the chain length (0 = read from the run JSON)")
    ap.add_argument("--nn-ranks", type=int, nargs="+", default=[0, 1, 2, 5, 10],
                    help="fuzzy: query with the k-th nearest embedding neighbour (0 = exact)")
    ap.add_argument("--settle-beta", type=float, default=0.0,
                    help="fuzzy: override the read settle sharpness (0 = model default); "
                         "lower = softer match = wider fuzzy basin")
    ap.add_argument("--store-grow", action="store_true",
                    help="context/sparse: grow the store like a KV cache (no ring lapping, r100)")
    ap.add_argument("--rare-keys", action="store_true",
                    help="sparse: draw fact keys from the RAREST words (low prose collision, r102)")
    ap.add_argument("--run", type=Path, required=True, help="train.py --out JSON")
    ap.add_argument("--ckpt", type=Path, required=True, help="train.py --save state_dict")
    ap.add_argument("--run-b", type=Path, default=None,
                    help="swap: foreign model's --out JSON (for the transplant test)")
    ap.add_argument("--ckpt-b", type=Path, default=None,
                    help="swap: foreign model's --save state_dict")
    ap.add_argument("--dir", type=Path, default=Path("persist_store"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--mismatch", action="store_true",
                    help="read: load another batch's store for each query (control)")
    ap.add_argument("--mem-list", type=int, nargs="+", default=[512],
                    help="context: store sizes M to sweep")
    ap.add_argument("--nseg-list", type=int, nargs="+", default=[4, 8, 16, 32],
                    help="context: segment counts (bridge = n_seg-1 windows) to sweep")
    ap.add_argument("--policies", nargs="+", default=["ring", "oracle"],
                    choices=["ring", "oracle", "surprisal", "surprisal_dedup"],
                    help="evict: eviction policies to compare")
    ap.add_argument("--topk-list", type=int, nargs="+", default=[1, 2, 4, 8, 0],
                    help="sparse: top-k values to read with (0 = dense graded baseline)")
    ap.add_argument("--shuffle", action="store_true",
                    help="sparse/passkey: matched control — read a neighbour's store (-> chance)")
    ap.add_argument("--depth-list", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0],
                    help="passkey: needle depths (fraction of the context) to sweep")
    ap.add_argument("--pool-modes", nargs="+", default=["none", "token", "ann"],
                    choices=["none", "token", "ann"],
                    help="pool: candidate-pool indices to compare")
    ap.add_argument("--pool-size", type=int, default=16,
                    help="pool: P / lsh granularity target")
    ap.add_argument("--pool-anchor", type=int, default=None,
                    help="pool: build the pool at this read layer and REUSE it at "
                         "later read layers (cross-layer reuse); -1/omit = per-layer")
    ap.add_argument("--quant-bits", type=int, nargs="+", default=[16, 8, 4, 3, 2],
                    help="quant: bit widths to fake-quantize the store to (16 = off)")
    ap.add_argument("--quant-what", default="both", choices=["keys", "values", "both"],
                    help="quant: which stored codes to quantize")
    a = ap.parse_args()
    a.mem_list = a.mem_list
    a.nseg_list = a.nseg_list
    device = "cuda" if torch.cuda.is_available() else "cpu"
    {"write": write, "read": read, "taxonomy": taxonomy,
     "context": context, "evict": evict, "sparse": sparse, "pool": pool,
     "quant": quant, "swap": swap, "chain_ar": chain_ar, "fuzzy": fuzzy,
     "semantic": semantic, "passkey": passkey}[a.mode](a, device)


if __name__ == "__main__":
    main()
