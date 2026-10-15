"""
train.py — segmented language-model training, with or without the store.

    uv run --group sleep python -m mech.train --hc off --steps 2000
    uv run --group sleep python -m mech.train --hc on  --steps 2000

The two must be run with identical everything else: same seed, same batches,
same schedule. HC-off is the control that distinguishes "the layer does not
help" from "the model is too small", and the zero-init read gate makes it an
exact control rather than an approximate one (tests_mech).

Loss is reported PER SEGMENT INDEX as well as in aggregate. That split is the
measurement that matters here: segment 0 has no history to recall, so memory
cannot help there, and any real effect has to show up as a widening gap on
later segments.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from mech.config import MemConfig
from mech.data import BookCorpus, load, load_books, sample
from mech.bpe import BPETokenizer, load_books_bpe
from mech.hippo_layer import HippoMemory
from mech.model import GPTConfig, TinyGPT
from mech.tasks import (N_SYMBOLS, TASK_CHARS, answer_stats, bpe_chain_batch,
                        bpe_collision_batch, bpe_hop_batch, bpe_recall_batch,
                        bpe_semchain_batch, bpe_update_batch, exact_match, make_batch,
                        mqar_batch, mqar_vocab)


def build(args, vocab_size, device):
    gcfg = GPTConfig(vocab_size=vocab_size, d_model=args.d_model,
                     n_layers=args.n_layers, n_heads=args.n_heads,
                     seg_len=args.seg_len)
    mem = None
    if args.hc == "on":
        d_mem = args.d_model if not args.no_write_identity else args.d_mem
        rl = tuple(args.read_layers)
        if getattr(args, "shared_per_layer", False) and len(rl) <= 1:
            rl = tuple(range(args.n_layers))   # r103: every layer reads+writes
        mcfg = MemConfig(d_model=args.d_model, d_mem=d_mem,
                         write_identity=not args.no_write_identity,
                         dg_bypass=args.dg_bypass,
                         memory_size=args.memory_size, readout=args.readout,
                         store_grow=getattr(args, "store_grow", False),
                         read_layers=rl,
                         shared_per_layer=getattr(args, "shared_per_layer", False),
                         shared_pl_key=getattr(args, "shared_pl_key", "token"),
                         write_layer=args.write_layer,
                         write_layers=(tuple(args.write_layers)
                                       if getattr(args, "write_layers", None) else None),
                         per_layer_store=getattr(args, "per_layer_store", False),
                         n_shards=getattr(args, "n_shards", 1),
                         key_layer=args.key_layer,
                         query_init=args.query_init,
                         key_pos_free=getattr(args, "key_pos_free", False),
                         freeze_query=getattr(args, "freeze_query", False),
                         value_skip_layer=getattr(args, "value_skip_layer", None),
                         value_skip_norm=getattr(args, "value_skip_norm", False),
                         value_shift=args.value_shift,
                         episode_evict=getattr(args, "episode_evict", False),
                         suppress_empty_read=getattr(args, "suppress_empty_read", False),
                         write_select=getattr(args, "write_select", "all"),
                         write_stride=getattr(args, "write_stride", 1),
                         write_frac=getattr(args, "write_frac", 1.0),
                         write_op_gate=getattr(args, "write_op_gate", False),
                         write_skip_pad=getattr(args, "write_skip_pad", False),
                         pad_id=getattr(args, "_pad", -1),
                         write_gate=not args.no_write_gate,
                         write_threshold=args.write_threshold,
                         shuffle_store=args.shuffle_store,
                         beta_grad=args.beta_grad,
                         read_grad_floor=args.read_grad_floor,
                         beta_max=args.beta_max,
                         beta=getattr(args, "beta", 8.0),
                         recency_read=getattr(args, "recency_read", False),
                         recency_tau=getattr(args, "recency_tau", 0.3),
                         fine_key=getattr(args, "fine_key", False),
                         fine_key_alpha=getattr(args, "fine_key_alpha", 1.0),
                         hop_key=getattr(args, "hop_key", "token"),
                         hop_reindex=getattr(args, "hop_reindex", False),
                         hop_grad=getattr(args, "hop_grad", False),
                         hop_ptr_beta=getattr(args, "hop_ptr_beta", 8.0),
                         loop_read=getattr(args, "loop_read", False),
                         loop_iters=getattr(args, "loop_iters", 2),
                         loop_layer=getattr(args, "loop_layer", None),
                         n_steps=args.n_steps)
        mem = HippoMemory(mcfg)
    return TinyGPT(gcfg, mem).to(device)


DIAG_OFFSETS = (-1, 0, 1)
DIAG_KEYS = ("n", "top1", "mass", "match_ok_ans_ok", "match_ok_ans_bad",
             "match_bad_ans_ok", "match_bad_ans_bad",
             *(f"top1@{d:+d}" for d in DIAG_OFFSETS),
             *(f"mass@{d:+d}" for d in DIAG_OFFSETS))


def slot_diagnostic(model, x, y, logits, mask, diag) -> None:
    """Attribute each MQAR answer to MATCHING or to CONTENT.

    The facts are tiled through segment 0, which the store writes as slots
    0..S-1-shift with slot p holding key h_p and value h_{p+shift}. A slot is
    CORRECT if its key position carries the queried key token, so its value is
    the answer. Crossing "did the read land on a correct slot" with "was the
    answer right" separates the two failure modes this project keeps meeting:
      right slot, wrong answer  -> content lost between store and output
                                   (the 40% "Mittens for Miso" fusion result)
      wrong slot, wrong answer  -> matching failed (the correlated-key cliff)
    Only valid for the unscrambled store: a shuffled store reads another
    example's slots, so this metric is meaningless there by construction.
    """
    head = next(iter(model.memory.read_heads.values()))
    w = (head.stash or {}).get("w")
    if w is None:
        return
    B, n_seg, S = x.shape
    last = n_seg - 1
    bidx = torch.arange(B, device=x.device)
    pos = mask[:, last].float().argmax(dim=1)          # answer position, [B]
    wq = w[bidx, pos]                                  # retrieval weights, [B, M]
    qkey = x[bidx, last, pos]                          # the queried key token
    n_key = S - model.memory.cfg.value_shift
    arg = wq.argmax(dim=1)

    def aligned(d: int) -> torch.Tensor:
        """Slots p whose position p+d carries the queried key token. d=0 is an
        identity slot ("this is k_j"). d=+1 is the slot one BEFORE the key,
        whose state at the last block predicts k_j — r2 measured a last-block
        write answering 44% while landing on d=0 slots 0% of the time, so
        a single assumed alignment misattributes it."""
        c = torch.zeros_like(wq, dtype=torch.bool)
        lo, hi = max(0, -d), min(n_key, S - d)
        c[:, lo:hi] = x[:, 0, lo + d:hi + d] == qkey[:, None]
        return c

    for d in DIAG_OFFSETS:
        cd = aligned(d)
        diag[f"top1@{d:+d}"] += int(cd[bidx, arg].sum())
        diag[f"mass@{d:+d}"] += float((wq * cd).sum())

    correct = aligned(0)
    top1 = correct[bidx, arg]
    ans = logits[bidx, pos].argmax(-1) == y[bidx, last, pos]
    diag["n"] += B
    diag["top1"] += int(top1.sum())
    diag["mass"] += float((wq * correct).sum())
    diag["match_ok_ans_ok"] += int((top1 & ans).sum())
    diag["match_ok_ans_bad"] += int((top1 & ~ans).sum())
    diag["match_bad_ans_ok"] += int((~top1 & ans).sum())
    diag["match_bad_ans_bad"] += int((~top1 & ~ans).sum())


def diag_applies(model, seg_len: int) -> bool:
    """Is the slot diagnostic meaningful for this configuration?

    It maps slot p to position p of segment 0, which holds only when every
    position is written, in order, into a store big enough for them. Three
    things break that, and all three now exist:
      write_select != "all"  -> only some positions earn a slot (r30)
      episode_evict          -> allocation is by policy, not in order (r31)
      memory_size < S-shift  -> the store cannot hold one segment; indexing the
                                diagnostic past the store CRASHED r31's M=64 runs
    """
    cfg = model.memory.cfg
    return (cfg.write_per_token
            and getattr(cfg, "write_select", "all") == "all"
            and not getattr(cfg, "episode_evict", False)
            and cfg.memory_size >= seg_len - cfg.value_shift)


def run_example(model, x, y, device, mask=None, answer_weight: float = 0.0,
                diag: dict | None = None, stats: dict | None = None):
    """One batched example = n_seg consecutive segments, store carried across.

    Returns mean loss, per-segment losses, and (correct, total) on the answer
    positions when a mask is supplied. `diag`, if given, accumulates the
    matching-vs-content attribution on the final (query) segment.
    """
    B, n_seg, _ = x.shape
    if model.memory is not None:
        model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
    per_seg, loss_sum, hits, tot = [], None, 0, 0
    for s in range(n_seg):
        if model.memory is not None and model.memory.cfg.episode_evict:
            # One segment = one episode. Segment 0 carries the facts (an event
            # WITH its response: k->v pairs), so it is complete and must survive
            # eviction; distractor segments are incomplete and go first.
            model.memory.store.begin_episode(complete=(s == 0))
        logits, h_write = model(x[:, s])
        flat = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(flat, y[:, s].reshape(-1))
        per_seg.append(float(loss.detach()))
        if mask is not None and mask[:, s].any():
            # The answer is 2 characters out of 512 positions, so inside a plain
            # LM loss it is ~0.4% of the gradient and the model simply ignores
            # it — measured as a 6.9% ceiling even with the facts in context.
            # Weight the answer positions explicitly; this is the "mixed LM +
            # task loss" the design calls for.
            sel = mask[:, s].reshape(-1)
            if answer_weight > 0:
                loss = loss + answer_weight * F.cross_entropy(
                    flat[sel], y[:, s].reshape(-1)[sel])
            h, t = exact_match(logits.detach(), y[:, s], mask[:, s])
            hits += h; tot += t
            if stats is not None:
                # per-character accuracy and answer CE alongside exact match,
                # because exact match at char level is too coarse to see a real
                # but partial binding (see tasks.answer_stats)
                answer_stats(logits.detach(), y[:, s], mask[:, s], stats)
            if (diag is not None and s == n_seg - 1 and model.memory is not None
                    and diag_applies(model, x.shape[2])):
                slot_diagnostic(model, x, y, logits.detach(), mask, diag)
        loss_sum = loss if loss_sum is None else loss_sum + loss
        if model.memory is not None and h_write is not None:
            model.memory.commit(h_write.detach(),    # commit at the boundary
                                score=getattr(model, "write_score", None))
            if model.memory.cfg.episode_evict:
                model.memory.store.end_episode()
    return loss_sum / n_seg, per_seg, (hits, tot)


@torch.no_grad()
def evaluate(model, ids, tok, args, device, iters=20):
    model.eval()
    gen = torch.Generator().manual_seed(1234)        # same eval batches always
    agg, seg_acc, hits, tot = [], None, 0, 0
    stats: dict = {}
    diag = None
    if (model.memory is not None and args.task == "mqar"
            and not args.facts_in_context and not args.shuffle_store):
        diag = dict.fromkeys(DIAG_KEYS, 0)
        for head in model.memory.read_heads.values():
            head.stash = {}
    for _ in range(iters):
        x, y, m = get_batch(ids, tok, args, device, gen)
        loss, per_seg, (h, t) = run_example(model, x, y, device, m,
                                            args.answer_weight, diag, stats)
        agg.append(float(loss)); hits += h; tot += t
        seg_acc = per_seg if seg_acc is None else [a + b for a, b in zip(seg_acc, per_seg)]
    if model.memory is not None:
        for head in model.memory.read_heads.values():
            head.stash = None
    model.train()
    acc = hits / tot if tot else None
    if diag and diag["n"]:
        n = diag.pop("n")
        diag = {"top1": diag.pop("top1") / n, "mass": diag.pop("mass") / n,
                **{k: v / n for k, v in diag.items()}}
    else:
        diag = None
    if stats.get("char_total"):
        per_pos = {f"pos{k}": stats[f"pos{k}_hits"] / stats[f"pos{k}_total"]
                   for k in range(8) if stats.get(f"pos{k}_total")}
        stats = {"char_acc": stats["char_hits"] / stats["char_total"],
                 "answer_ce": stats["ce_sum"] / stats["ce_n"], **per_pos}
    else:
        stats = None
    return sum(agg) / len(agg), [s / iters for s in seg_acc], acc, diag, stats


def get_batch(ids, tok, args, device, gen):
    """LM batches, or recall batches. Returns x, y, answer_mask|None."""
    if args.task == "lm":
        if isinstance(ids, BookCorpus):        # PG-19: one example, one book
            x, y = ids.sample(args.batch, args.n_seg, args.seg_len, device, gen)
        else:
            x, y = sample(ids, args.batch, args.n_seg, args.seg_len, device, gen)
        return x, y, None
    if args.task == "mqar":
        return mqar_batch(ids, tok.vocab_size, args.batch, args.n_seg,
                          args.seg_len, args.n_facts, device, gen,
                          facts_in_context=args.facts_in_context,
                          tile=not getattr(args, "no_tile", False),
                          n_symbols=getattr(args, "n_symbols", N_SYMBOLS))
    if args.task == "bpe_recall":
        return bpe_recall_batch(ids, args.batch, args.n_seg, args.seg_len,
                                args.n_facts, args._key_pool, args._val_pool,
                                args._qtok, args._pad, device, gen,
                                facts_in_context=args.facts_in_context,
                                tile=not getattr(args, "no_tile", False),
                                needle_seg=getattr(args, "needle_seg", 0))
    if args.task == "bpe_update":
        return bpe_update_batch(ids, args.batch, args.n_seg, args.seg_len,
                                args.n_facts, args._key_pool, args._val_pool,
                                args._qtok, args._pad, device, gen,
                                query_updated=not getattr(args, "query_nonupdated",
                                                          False))
    if args.task == "bpe_hop":
        return bpe_hop_batch(ids, args.batch, args.n_seg, args.seg_len,
                             args.n_facts, args._key_pool, args._val_pool,
                             args._qtok, args._pad, device, gen,
                             n_hops=getattr(args, "n_hops", 2),
                             broken=getattr(args, "broken_chain", False),
                             n_chains=getattr(args, "n_chains", 1))
    if args.task == "bpe_chain":
        return bpe_chain_batch(ids, args.batch, args.n_seg, args.seg_len,
                               args.n_facts, args._key_pool, args._val_pool,
                               args._qtok, args._pad, device, gen,
                               n_hops=getattr(args, "n_hops", 2),
                               broken=getattr(args, "broken_chain", False),
                               n_chains=getattr(args, "n_chains", 4))
    if args.task == "bpe_collision":
        return bpe_collision_batch(ids, args.batch, args.n_seg, args.seg_len,
                                   args.n_facts, args._key_pool, args._val_pool,
                                   args._qtok, args._pad, device, gen,
                                   collisions_per_key=getattr(args, "collisions_per_key", 4),
                                   scramble_context=getattr(args, "scramble_context", False))
    if args.task == "bpe_semchain":
        return bpe_semchain_batch(ids, args.batch, args.n_seg, args.seg_len,
                                  args.n_facts, args._lexicon, args._val_pool,
                                  args._qtok, args._pad, device, gen,
                                  broken=getattr(args, "broken_chain", False))
    return make_batch(ids, tok, args.batch, args.n_seg, args.seg_len,
                      args.n_facts, device, gen,
                      facts_in_context=args.facts_in_context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hc", choices=["on", "off"], default="off")
    ap.add_argument("--task", choices=["lm", "recall", "mqar", "bpe_recall",
                                       "bpe_update", "bpe_hop", "bpe_chain",
                                       "bpe_collision", "bpe_semchain"], default="lm")
    ap.add_argument("--lexicon-size", type=int, default=64,
                    help="bpe_semchain: size of the fixed syn->canon lexicon (learned in weights)")
    ap.add_argument("--store-grow", action="store_true",
                    help="r100: grow the store like a KV cache (append) instead of a fixed ring")
    ap.add_argument("--n-hops", type=int, default=2,
                    help="bpe_hop: chain length (query head, target tail n hops away)")
    ap.add_argument("--broken-chain", action="store_true",
                    help="bpe_hop: cut the last link so head cannot reach tail (control)")
    ap.add_argument("--n-chains", type=int, default=1,
                    help="bpe_hop: >1 = leak-proof (several shuffled chains, query a random one)")
    ap.add_argument("--write-op-gate", action="store_true",
                    help="r76: gated memory ops — one slot per distinct key (new/update/skip)")
    ap.add_argument("--write-skip-pad", action="store_true",
                    help="r84: don't store pad-valued slots (removes the key-and-value collision)")
    ap.add_argument("--recency-read", action="store_true",
                    help="r67: bias the read toward the slot written last, so the "
                         "newest value of a re-stated key wins (both stay stored)")
    ap.add_argument("--recency-tau", type=float, default=0.3,
                    help="strength of the recency bias added to cos sims")
    ap.add_argument("--fine-key", action="store_true",
                    help="r89: position-free BIGRAM key emb(t)+alpha*emb(t-1) — a fine "
                         "context term that separates facts sharing a key token (DG)")
    ap.add_argument("--fine-key-alpha", type=float, default=1.0,
                    help="weight of the fine (previous-token) term in the bigram key")
    ap.add_argument("--collisions-per-key", type=int, default=4,
                    help="bpe_collision: C facts sharing each key token (chance = 1/C)")
    ap.add_argument("--scramble-context", action="store_true",
                    help="bpe_collision: query with the WRONG context token (control)")
    ap.add_argument("--hop-key", choices=["token", "retrieved", "residual", "pointer"],
                    default="token", help="r68/r75: multi-hop chaining key source")
    ap.add_argument("--hop-ptr-beta", type=float, default=8.0,
                    help="r75: sharpness of the pointer decode onto the token manifold")
    ap.add_argument("--loop-read", action="store_true",
                    help="r82: loop-gate latent chaining (iterate the read layer, hard-decode between)")
    ap.add_argument("--loop-iters", type=int, default=2, help="r82: fixed loop hops")
    ap.add_argument("--loop-layer", type=int, default=None, help="r82: which read layer loops")
    ap.add_argument("--hop-reindex", action="store_true",
                    help="r73: learned map re-landing a retrieval as the next hop's key")
    ap.add_argument("--hop-grad", action="store_true",
                    help="r74: keep the graph across hops (end-to-end gradient through the chain)")
    ap.add_argument("--query-nonupdated", action="store_true",
                    help="bpe_update: query a fact that was NOT updated (control)")
    ap.add_argument("--n-facts", type=int, default=8)
    ap.add_argument("--episode-evict", action="store_true",
                    help="evict by EPISODE, not by ring order: incomplete episodes\n"
                         "                        go before complete ones (the user's rule)")
    ap.add_argument("--write-select", default="all",
                    choices=["all", "stride", "surprisal", "last"],
                    help="which positions earn a store slot. Causal attention makes\n"
                         "                        one-slot-per-token redundant; this is the write RATE knob")
    ap.add_argument("--write-stride", type=int, default=1,
                    help="--write-select stride: keep every k-th position")
    ap.add_argument("--write-frac", type=float, default=1.0,
                    help="--write-select surprisal: keep this fraction, the most\n"
                         "                        surprising by the model's own NLL")
    ap.add_argument("--n-symbols", type=int, default=N_SYMBOLS,
                    help="distinct key (and value) symbols; chance = 1/n. The\n"
                         "                          default 32 also caps n_facts at 32, so raising this is\n"
                         "                          what separates fact count from key vocabulary")
    ap.add_argument("--answer-weight", type=float, default=1.0,
                    help="extra CE weight on the answer positions")
    ap.add_argument("--no-tile", action="store_true",
                    help="mqar: each fact appears ONCE in segment 0 (rest is text),\n                          so the answer cannot be predicted from repeats")
    ap.add_argument("--facts-in-context", action="store_true",
                    help="ceiling control: paste the facts into the query segment")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--suppress-empty-read", action="store_true",
                    help="skip retrieval while the store is empty. r34 measured\n                          the cost of reading an empty store at segment 0")
    ap.add_argument("--data-books", default=None,
                    help="PG-19 root holding <root>/test/*.txt (training corpus)\n                          and <root>/validation/*.txt (held out). Overrides --data;\n                          keeps books apart so no example straddles two of them.")
    ap.add_argument("--max-books", type=int, default=None,
                    help="cap books per split, for quick runs")
    ap.add_argument("--bpe", default=None,
                    help="path to a BPE tokenizer .json; trained from --data-books\n                          if absent. Switches --data-books to token-level.")
    ap.add_argument("--bpe-vocab", type=int, default=8192,
                    help="BPE vocab size when training the tokenizer")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1,
                    help="micro-batches per optimiser step (effective batch =\n                          batch x accum), for stores too large for a full batch")
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--seg-len", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-mem", type=int, default=128)
    ap.add_argument("--memory-size", type=int, default=64)
    ap.add_argument("--readout", default="graded")
    ap.add_argument("--beta-grad", type=float, default=1.0,
                    help="backward-pass temperature. The forward uses the\n                          sharp weights; too large a gap and the surrogate\n                          gradient describes a computation the model is not\n                          performing, which bites harder as keys separate")
    ap.add_argument("--beta-max", type=float, default=64.0)
    ap.add_argument("--beta", type=float, default=8.0,
                    help="initial forward temperature of the read. With --n-steps 0\n                          there is no anneal, so this is the only sharpness")
    ap.add_argument("--read-grad-floor", type=float, default=0.0,
                    help="gradient into the read path that does not change the\n                          forward; breaks the shut-gate trap where alpha~0\n                          starves the query projection. 0 = off")
    ap.add_argument("--n-steps", type=int, default=3,
                    help="CA3 settling iterations; 0 = one-shot nearest\n                          neighbour, i.e. no attractor dynamics")
    ap.add_argument("--flat-lr", action="store_true",
                    help="constant LR instead of OneCycle")
    ap.add_argument("--read-layers", type=int, nargs="+", default=[1])
    ap.add_argument("--write-layer", type=int, default=4)
    ap.add_argument("--write-layers", type=int, nargs="+", default=None,
                    help="r56: store the MEAN of the hidden states at these layers "
                         "(write at every layer) instead of a single --write-layer")
    ap.add_argument("--per-layer-store", action="store_true",
                    help="r57: a unique store per layer in --write-layers, keyed and "
                         "valued at that layer (removes the single layer-0 key)")
    ap.add_argument("--n-shards", type=int, default=1,
                    help="r109: shard facts across N shared stores by key token (tok mod N); "
                         "each shard holds ~1/N of the keys. 1 = single shared store.")
    ap.add_argument("--shared-per-layer", action="store_true",
                    help="r103: per-layer READ and WRITE into the ONE shared store "
                         "(replaces read-early/write-late). Auto-sets read_layers to "
                         "every block unless --read-layers gives more than one.")
    ap.add_argument("--shared-pl-key", choices=["token", "residual"], default="token",
                    help="r103 key source for --shared-per-layer: shared token key, or "
                         "each layer's own pre-read residual")
    ap.add_argument("--key-layer", type=int, default=None,
                    help="take the KEY from this block (pre-read) instead of the\n                          write block; the value still comes from --write-layer")
    ap.add_argument("--value-skip-layer", type=int, default=None,
                    help="add this block's output to the write-block state to form\n                          the value (a skip connection into the store)")
    ap.add_argument("--value-skip-norm", action="store_true",
                    help="with --value-skip-layer: unit-normalise both halves before\n                          summing so the last block cannot swamp the skip")
    ap.add_argument("--freeze-query", action="store_true",
                    help="hold the query projection at its init (use with\n                          --query-init identity)")
    ap.add_argument("--key-pos-free", action="store_true",
                    help="with --key-layer 0 --read-layers 0: key and query from the\n                          token embedding only, no position embedding")
    ap.add_argument("--query-init", choices=["random", "identity", "key_proj"],
                    default="random",
                    help="identity: query projection starts as I, so a layer-1\n                          query and a layer-1 key begin in the same space.\n                          key_proj: copy the write head's frozen key projection —\n                          the same argument when d_mem != d_model makes identity\n                          impossible (narrow stores)")
    ap.add_argument("--value-shift", type=int, default=1,
                    help="value = state at t+shift. 1 suits a mid-stack 'what is\n                          here' state; 0 suits the last block, which already\n                          encodes 'what comes next'")
    ap.add_argument("--write-threshold", type=float, default=0.6)
    ap.add_argument("--no-write-gate", action="store_true")
    ap.add_argument("--dg-bypass", action="store_true",
                    help="skip top-k sparsification of the query; with raw\n                          hidden states as keys a sparse query matches dense\n                          keys poorly, and v0.1 measured DG separation\n                          hurting semantic matching anyway")
    ap.add_argument("--no-write-identity", action="store_true",
                    help="use the frozen projections instead of raw states")
    ap.add_argument("--shuffle-store", action="store_true",
                    help="control: read another example's memories")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--init", type=Path, default=None,
                    help="load weights from a --save state_dict before training\n                          (store size may differ; architecture must match)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save", type=Path, default=None,
                    help="save the final state_dict here, so a finished run can be\n                          re-diagnosed without retraining")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.task in ("bpe_recall", "bpe_update", "bpe_hop", "bpe_chain",
                     "bpe_collision", "bpe_semchain"):
        if not args.data_books:
            ap.error(f"--task {args.task} needs --data-books (PG-19 root)")
        if args.bpe and Path(args.bpe).exists():
            tok = BPETokenizer.load(args.bpe)
            print(f"bpe: loaded {args.bpe}, vocab {tok.vocab_size}", flush=True)
        else:
            from mech.bpe import load_books_bpe as _lbb  # noqa
            root = Path(args.data_books)
            texts = [pp.read_text(encoding="utf-8", errors="replace")
                     for pp in sorted((root / "test").glob("*.txt"))
                     [:(args.max_books or 10**9)]]
            tok = BPETokenizer.train(texts, vocab_size=args.bpe_vocab)
            save_to = args.bpe or f"{args.data_books}/bpe_{args.bpe_vocab}.json"
            tok.save(save_to)
            print(f"bpe: trained on {len(texts)} books, vocab {tok.vocab_size}, "
                  f"saved {save_to}", flush=True)
        train_ids, val_ids = load_books_bpe(args.data_books, tok,
                                            max_books=args.max_books)
        # partition single-token words into disjoint key / value pools + a query
        # marker, so a key is one real word competing with the prose vocabulary
        words = tok.single_token_words()
        need = (3 * args.lexicon_size + 1 if args.task == "bpe_semchain"
                else 4 * args.n_facts + 1 if args.task == "bpe_collision"
                else 2 * args.n_facts + (2 if args.task == "bpe_update" else 1))
        if len(words) < need:
            ap.error(f"only {len(words)} single-token words; need {need}")
        half = len(words) // 2
        args._key_pool = words[:half]
        args._val_pool = words[half:2 * half]
        args._qtok = words[-1]                     # reserved query marker
        args._pad = tok.pad_id
        if args.task == "bpe_semchain":
            L = args.lexicon_size                  # FIXED syn->canon lexicon (learned in weights)
            args._lexicon = list(zip(words[:L], words[L:2 * L]))   # (syn, canon)
            args._val_pool = words[2 * L:3 * L]                    # disjoint value pool
            print(f"bpe_semchain: fixed lexicon of {len(args._lexicon)} syn->canon pairs, "
                  f"{len(args._val_pool)} values", flush=True)
        print(f"bpe_recall: {len(args._key_pool)} keys / {len(args._val_pool)} "
              f"values, prose {train_ids.numel()/1e6:.1f}M / "
              f"{val_ids.numel()/1e6:.1f}M tokens", flush=True)
    elif args.data_books:
        tok, train_ids, val_ids, dropped = load_books(
            args.data_books, extra_chars=TASK_CHARS, max_books=args.max_books)
        print(f"pg19: {len(train_ids)} train books ({train_ids.n_tokens/1e6:.1f}M "
              f"tokens), {len(val_ids)} held-out books "
              f"({val_ids.n_tokens/1e6:.1f}M), vocab {tok.vocab_size}, "
              f"{dropped*100:.4f}% of held-out chars outside the train vocab",
              flush=True)
        if args.task != "lm":
            # Books stay separate only for the LM task; the recall tasks use
            # prose as independent per-segment filler, so a flat stream is both
            # correct and what their `ids[start:start + seg_len]` expects.
            train_ids, val_ids = train_ids.flat(), val_ids.flat()
            print(f"pg19: flattened for --task {args.task} (distractor prose)",
                  flush=True)
    else:
        tok, train_ids, val_ids = load(args.data, extra_chars=TASK_CHARS)
    vocab = (mqar_vocab(tok.vocab_size, getattr(args, "n_symbols", N_SYMBOLS))
             if args.task == "mqar" else tok.vocab_size)
    model = build(args, vocab, device)
    if getattr(args, "init", None):
        # Warm start. Store tensors are non-persistent buffers, so a checkpoint
        # trained at one memory_size loads into a model with another; only the
        # learned weights (and the persistent centring mean) carry over.
        state = torch.load(args.init, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        # A newly-added, self-initialised module (e.g. the r73 reindex) is
        # legitimately absent from an older checkpoint — keep it at its init rather
        # than treating warm-start as a mismatch. Everything else must still match.
        NEW_OK = ("memory.reindex",)
        new_missing = [k for k in missing if any(k.startswith(t) for t in NEW_OK)]
        missing = [k for k in missing if k not in new_missing]
        # memory.tok_emb_ref is a shared alias of tok_emb.weight that older code
        # accidentally registered into the state_dict; it is not a real parameter, so
        # a checkpoint carrying it is fine to load (the alias is re-bound each forward).
        unexpected = [k for k in unexpected if not k.startswith("memory.tok_emb_ref")]
        if missing or unexpected:
            raise SystemExit(f"--init {args.init}: missing {missing}, unexpected {unexpected}")
        note = f" (+kept at init: {new_missing})" if new_missing else ""
        print(f"initialised from {args.init}{note}")
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1)
    sched = (torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=1)
             if args.flat_lr else
             torch.optim.lr_scheduler.OneCycleLR(
                 opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"hc={args.hc}  vocab={vocab}  params={model.n_params()/1e6:.2f}M  "
          f"trainable={trainable/1e6:.2f}M  device={device}")
    print(f"example = {args.n_seg} x {args.seg_len} tokens, store carries across segments")

    gen = torch.Generator().manual_seed(args.seed)
    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        # --accum k: k micro-batches of --batch per optimiser step. Memory for a
        # store-carrying example scales with n_seg x memory_size (every
        # segment's [B,S,M] read tensors live until backward), so a large store
        # forces a small batch; r12 found that confounds store size with batch
        # size. Accumulation keeps the effective batch fixed.
        accum = getattr(args, "accum", 1)
        for _ in range(accum):
            x, y, m = get_batch(train_ids, tok, args, device, gen)
            loss, _, _ = run_example(model, x, y, device, m, args.answer_weight)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.steps:
            vloss, vseg, vacc, vdiag, vstat = evaluate(model, val_ids, tok, args, device)
            gates = model.memory.gate_magnitudes() if model.memory else {}
            per = "  ".join(f"s{i}:{v:.3f}" for i, v in enumerate(vseg))
            gtxt = ("  gates=" + ",".join(f"{k}:{v:.3f}" for k, v in gates.items())
                    if gates else "")
            atxt = f"  recall {100 * vacc:5.1f}%" if vacc is not None else ""
            occ = model.memory.mean_occupancy() if model.memory else None
            mtxt = f"  mem {occ:.0f}/{args.memory_size}" if occ is not None else ""
            # char accuracy and answer CE, because exact match at char level is
            # too coarse to see a partial binding (tasks.answer_stats)
            ptxt = (" " + "/".join(f"{100 * v:.0f}" for k, v in vstat.items()
                                   if k.startswith("pos")) if vstat else "")
            stxt = (f"  char {100 * vstat['char_acc']:5.1f}%{ptxt} "
                    f"ce {vstat['answer_ce']:.3f}" if vstat else "")
            dtxt = (f"  slot@1 {100 * vdiag['top1']:5.1f}% mass {vdiag['mass']:.2f} "
                    f"content-miss {100 * vdiag['match_ok_ans_bad']:4.1f}% "
                    f"match-miss {100 * vdiag['match_bad_ans_bad']:4.1f}% "
                    "align " + " ".join(f"{d:+d}:{100 * vdiag[f'top1@{d:+d}']:.0f}"
                                         for d in DIAG_OFFSETS)
                    if vdiag else "")
            print(f"step {step:>5}  train {float(loss.detach()):.4f}  val {vloss:.4f}  "
                  f"ppl {math.exp(vloss):.2f}{atxt}{mtxt}{stxt}{dtxt}  [{per}]{gtxt}")
            history.append({"step": step, "train": float(loss.detach()), "val": vloss,
                            "val_per_segment": vseg, "recall": vacc, "gates": gates,
                            "slots": vdiag, "answer": vstat})

    print(f"done in {time.time() - t0:.0f}s")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"-> {args.save}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"args": {k: (list(v) if isinstance(v, tuple) else str(v)
                          if isinstance(v, Path) else v)
                      for k, v in vars(args).items()},
             "history": history}, indent=2))
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
