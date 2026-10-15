"""
sleep_evict.py — sleep as the eviction mechanism, on the current architecture.

A slot is not dropped because it is old. It is dropped because its content now
lives in the weights. The loop, per r26:

  WAKE      weights frozen. Facts are written to the store and answered from it.
  DECODE    keys -> token ids by nearest neighbour against the embedding matrix.
            r26: exact (100% on fact symbols). Slot i+1's key is slot i's value
            token, so the key sequence IS the written text — no decoder needed.
  SLEEP     reads DISABLED, weights unfrozen, replay the decoded text. Reads are
            off on purpose: with retrieval live the model answers by looking up,
            the weights absorb nothing, and the store becomes a crutch that
            hides its own failure to transfer.
  VERIFY    probe with reads off and nothing in context. A correct answer there
            can only have come from the weights.
  FREE      reclaim only the slots VERIFY confirms. The rest wait for next sleep.

Three things this measures that the design needs and nobody has checked:

  * slots reclaimed per sleep (does eviction actually free anything?)
  * LM loss before/after (does consolidation damage general ability?)
  * **whether the surviving store still reads after sleep.** Keys were written
    by the pre-sleep weights. If sleep moves the key-producing layer, every slot
    not yet consolidated may become unreadable — consolidation would destroy the
    memory it is meant to drain. This is the loop's real risk and it is reported
    as `post_sleep_store_recall`.

    uv run --group sleep python -m mech.sleep_evict --run RESULT.json --ckpt CKPT.pt
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from mech.data import sample
from mech.decode import decode_emb
from mech.persist import answer_hits, restore
from mech.train import get_batch, run_example


@torch.no_grad()
def wake(model, x, device):
    """Fill the store from every segment but the last, weights untouched."""
    n_seg = x.shape[1]
    model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
    for s in range(n_seg - 1):
        # one segment = one episode, tracked even when eviction is the plain ring:
        # the ownership is what lets re-encoding replay the original selection
        model.memory.store.begin_episode(complete=(s == 0))
        _, h_write = model(x[:, s])
        model.memory.commit(h_write, score=getattr(model, "write_score", None))
        model.memory.store.end_episode()


@torch.no_grad()
def answer(model, x, y, m, *, read: bool):
    prev = model.memory.read_enabled
    model.memory.set_read(read)
    logits, _ = model(x)
    model.memory.set_read(prev)
    return answer_hits(logits, y, m)


@torch.no_grad()
def decode_store(model, n_slots):
    """Keys -> token ids, in slot order: the transcript of what was written."""
    keys = model.memory.store.keys[:, :n_slots]
    return decode_emb(keys, model.tok_emb.weight)


@torch.no_grad()
def share_facts(x, y, m, n_facts, qtok):
    """Give every example in the batch the SAME facts (example 0's).

    MQAR draws a fresh fact set per example, so across a batch one key maps to
    several different values. Weights cannot hold contradictions, which makes
    consolidation ill-posed at batch > 1 — measured: naive sleep over 64
    independent fact sets consolidated 3.1% while doubling LM loss. One "user"
    per sleep is the honest unit (wake_sleep.py used one persona for the same
    reason); sharing the facts keeps many queries per fact for the eval.
    """
    B, n_seg, S = x.shape
    pairs = x[0, 0, :2 * n_facts].clone()
    x = x.clone()
    x[:, 0, :2 * n_facts] = pairs                      # same facts, own distractors
    ks, vs = pairs[0::2], pairs[1::2]
    # re-ask each example's query against the shared set, keeping its own index
    asked = x[:, n_seg - 1, S - 2]
    for b in range(B):
        j = b % n_facts
        x[b, n_seg - 1, S - 2] = ks[j]
        x[b, n_seg - 1, S - 1] = vs[j]
    y = torch.zeros_like(x)
    y[:, :, :-1] = x[:, :, 1:]
    return x, y, m


@torch.no_grad()
def held_out_query(x, n_facts, generator):
    """A query about a DIFFERENT fact of the same example.

    Sleep replays the transcript *and* the query it answers, so verifying on that
    same query would be training on the test — it would score 100% by rote. The
    query segment ends `[<query>, k_j, v_j]`, and (untiled) segment 0 opens with
    the pairs, so swapping in another pair from the same example gives a fact the
    weights were never supervised on, only replayed.
    """
    xq = x[:, -1].clone()
    B, S = xq.shape
    pairs = x[:, 0, :2 * n_facts]                      # k1 v1 k2 v2 ...
    asked = xq[:, S - 2]
    for b in range(B):
        ks = pairs[b, 0::2]
        same = (ks == asked[b]).nonzero()
        cur = int(same[0]) if len(same) else 0
        j = (cur + 1) % max(len(ks), 1)                # the next fact along
        xq[b, S - 2] = ks[j]
        xq[b, S - 1] = pairs[b, 2 * j + 1]
    yq = torch.zeros_like(xq)
    yq[:, :-1] = xq[:, 1:]
    mq = torch.zeros_like(xq, dtype=torch.bool)
    mq[:, S - 2] = True
    return xq, yq, mq


@torch.no_grad()
def lm_loss(model, ids, args, device, iters=10):
    """Held-out LM loss — catches consolidation damaging general ability."""
    prev = model.memory.read_enabled
    model.memory.set_read(False)
    g = torch.Generator().manual_seed(4242)
    tot = 0.0
    for _ in range(iters):
        xx, yy = sample(ids, args.batch, 1, args.seg_len, device, g)
        model.memory.reset(xx.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
        logits, _ = model(xx[:, 0])
        tot += float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     yy[:, 0].reshape(-1)))
    model.memory.set_read(prev)
    return tot / iters


@torch.no_grad()
def synth_queries(replay, n_facts, ids, tok, args, device, generator, qtok):
    """Build one supervised query per decoded fact, in the format the model answers.

    The first smoke test replayed the transcript as plain LM text and supervised
    only the single query the batch happened to ask. Result: the asked fact hit
    100% from weights, every OTHER fact 0% — "k v" appearing in text does not
    teach `<query> k -> v`. The store hands us every pair, so every pair gets a
    query, each with fresh distractor text so the weights learn the association
    rather than one sequence.
    """
    B = replay.shape[0]
    S = args.seg_len
    pairs = replay[:, :2 * n_facts]                     # untiled: pairs open seg 0
    xs = []
    for j in range(n_facts):
        seq = torch.zeros(B, S, dtype=torch.long, device=device)
        for b in range(B):
            start = int(torch.randint(len(ids) - S, (1,), generator=generator))
            seq[b] = ids[start:start + S].to(device)
        seq[:, S - 3] = qtok
        seq[:, S - 2] = pairs[:, 2 * j]
        seq[:, S - 1] = pairs[:, 2 * j + 1]
        xs.append(seq)
    x = torch.cat(xs, 0)
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    m = torch.zeros_like(x, dtype=torch.bool)
    m[:, S - 2] = True
    return x, y, m


def sleep(model, replay, supervised, steps, lr, answer_weight, device,
          *, freeze_keys=False, rehearse=0.0, rehearse_mem=0.0, ids=None,
          args=None, tok=None, mem_batch=None):
    """Replay with reads OFF and the weights moving.

    Two protections, both prompted by the first smoke test, where naive sleep
    left the store unreadable (0%) and nearly doubled LM loss (0.89 -> 1.72):

    freeze_keys   the store's keys were written by the PRE-sleep embedding, and
                  matching is nearest-neighbour in that space. Let the embedding
                  move and every slot not yet consolidated becomes unreadable —
                  consolidation destroys the memory it is draining. Freezing the
                  token embedding and block 0 pins the key space.
    rehearse      interleave general LM batches, the standard guard against
                  catastrophic forgetting on a tiny replay set.
    rehearse_mem  interleave the MEMORY task itself, reads ON, on fresh facts.
                  r27 left one cost unsolved: after sleep the model writes new
                  memories worse (75% vs 100%), because LM rehearsal protects
                  general ability and nothing keeps the write path calibrated
                  against the read path. This rehearses exactly that.
    """
    x_s, y_s, m_s = supervised
    seg_len = x_s.shape[1]
    chunks = [replay[:, i:i + seg_len] for i in range(0, replay.shape[1], seg_len)]
    chunks = [c for c in chunks if c.shape[1] > 1]

    frozen = []
    if freeze_keys:
        frozen = [model.tok_emb.weight, model.pos_emb.weight, *model.blocks[0].parameters()]
        for p in frozen:
            p.requires_grad_(False)

    model.memory.set_read(False)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    g = torch.Generator().manual_seed(31337)
    for _ in range(steps):
        model.memory.reset(x_s.shape[0], device=device,
                           dtype=model.tok_emb.weight.dtype)
        logits, _ = model(x_s)                       # the facts, in query format
        flat = logits.reshape(-1, logits.size(-1))
        sel = m_s.reshape(-1)
        loss = answer_weight * F.cross_entropy(flat[sel], y_s.reshape(-1)[sel])
        loss = loss + F.cross_entropy(flat, y_s.reshape(-1))

        for c in chunks:                             # the transcript itself
            model.memory.reset(c.shape[0], device=device,
                               dtype=model.tok_emb.weight.dtype)
            lg, _ = model(c)
            loss = loss + F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)),
                                          c[:, 1:].reshape(-1)) / len(chunks)

        if rehearse_mem > 0 and ids is not None:     # keep the memory path calibrated
            # a memory-task example costs batch x n_seg x M, which at the 8x store
            # OOMs if it matches the sleep batch — it gets its own, smaller one
            b0 = args.batch
            if mem_batch:
                args.batch = mem_batch
            xm, ym, mm = get_batch(ids, tok, args, device, g)
            args.batch = b0
            model.memory.set_read(True)
            lm_, _, _ = run_example(model, xm, ym, device, mask=mm,
                                    answer_weight=answer_weight)
            model.memory.set_read(False)
            loss = loss + rehearse_mem * lm_

        if rehearse > 0 and ids is not None:         # guard general ability
            xx, yy = sample(ids, args.batch, 1, seg_len, device, g)
            model.memory.reset(xx.shape[0], device=device,
                               dtype=model.tok_emb.weight.dtype)
            lg, _ = model(xx[:, 0])
            loss = loss + rehearse * F.cross_entropy(
                lg.reshape(-1, lg.size(-1)), yy[:, 0].reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    for p in frozen:
        p.requires_grad_(True)
    model.eval()
    model.memory.set_read(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sleep-steps", type=int, default=400)
    ap.add_argument("--sleep-lr", type=float, default=1e-4)
    ap.add_argument("--freeze-keys", action="store_true",
                    help="pin the key space (token embedding + block 0) so the\n"
                         "                        store stays readable after sleep")
    ap.add_argument("--rehearse", type=float, default=0.0,
                    help="weight on interleaved general LM batches")
    ap.add_argument("--mem-batch", type=int, default=None,
                    help="batch for the memory-rehearsal example (default: --batch);\n"
                         "                        smaller keeps it in memory at large stores")
    ap.add_argument("--rehearse-mem", type=float, default=0.0,
                    help="weight on interleaved MEMORY-task batches (reads on,\n"
                         "                        fresh facts) — keeps the write path calibrated")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shared-facts", action="store_true",
                    help="one fact set for the whole batch — consolidation is\n"
                         "                        ill-posed when examples contradict each other")
    ap.add_argument("--reencode-rescore", action="store_true",
                    help="re-encode by re-running the write policy over the\n                          transcript (r32's behaviour, measured at 6.3%). The\n                          default rewrites the transcript 1:1, since the decoded\n                          store IS the selection — one token per slot.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    gen = torch.Generator().manual_seed(1234)
    x, y, m = get_batch(val_ids, tok, args, device, gen)
    if a.shared_facts:
        from mech.tasks import _sym
        qt = _sym(tok.vocab_size, getattr(args, "n_symbols", 32))[2]
        x, y, m = share_facts(x, y, m, args.n_facts, qt)
    query = (x[:, -1], y[:, -1], m[:, -1])

    # a query about a different fact of the same example — sleep never sees its
    # answer supervised, so it is the honest test of consolidation
    probe_q = held_out_query(x, args.n_facts, gen)

    # ---- WAKE: frozen weights, everything in the store ---------------------
    wake(model, x, device)
    n_slots = int(model.memory.store.n_stored[0])

    # ---- DECODE first: lm_loss() resets the store, so read it out now ------
    replay = decode_store(model, n_slots)
    shift = model.memory.cfg.value_shift
    S = args.seg_len
    truth = torch.cat([x[:, s, :S - shift] for s in range(x.shape[1] - 1)], dim=1)
    decode_exact = float((replay == truth[:, :n_slots]).float().mean())

    h, t = answer(model, *query, read=True)
    wake_recall = h / t
    h, t = answer(model, *probe_q, read=True)
    wake_recall_heldout = h / t
    h, t = answer(model, *probe_q, read=False)
    before_weights = h / t                 # what the weights knew already
    lm_before = lm_loss(model, val_ids, args, device)
    wake(model, x, device)                 # lm_loss cleared it; refill for later

    # ---- SLEEP: reads off, weights move ------------------------------------
    from mech.tasks import _sym
    qtok = _sym(tok.vocab_size, getattr(args, "n_symbols", 32))[2]
    supervised = synth_queries(replay, args.n_facts, val_ids, tok, args, device,
                               gen, qtok)
    pre = {k: v.detach().clone() for k, v in model.state_dict().items()}
    # sleep's own forward passes reset the store, so snapshot it: the question
    # "is the pre-sleep store still readable" is about KEY DRIFT, not about
    # whether sleep happened to clear the buffer
    snap = {f: getattr(model.memory.store, f).detach().clone()
            for f in model.memory.store.STATE}
    # non-persistent, but needed to replay the original selection (r32 fix)
    for f in ("ep_id", "slot_pos"):
        snap[f] = getattr(model.memory.store, f).detach().clone()
    sleep(model, replay, supervised, a.sleep_steps, a.sleep_lr,
          args.answer_weight, device, freeze_keys=a.freeze_keys,
          rehearse=a.rehearse, rehearse_mem=a.rehearse_mem, ids=val_ids,
          args=args, tok=tok, mem_batch=a.mem_batch)

    # ---- VERIFY: can the weights answer alone? -----------------------------
    h, t = answer(model, *probe_q, read=False)      # held-out fact
    after_weights = h / t
    h, t = answer(model, *query, read=False)        # the replayed fact (rote bound)
    after_weights_replayed = h / t
    lm_after = lm_loss(model, val_ids, args, device)

    # ---- the risk: does the surviving store still read post-sleep? ---------
    # keys in the store were written by the PRE-sleep weights
    for f, v in snap.items():
        getattr(model.memory.store, f).copy_(v)
    h, t = answer(model, *probe_q, read=True)
    post_sleep_store = h / t
    # and would a store written by the new weights read correctly?
    wake(model, x, device)
    h, t = answer(model, *probe_q, read=True)
    rewritten_store = h / t

    # RE-ENCODE: the repair for key drift. r26 showed the store decodes to exact
    # token ids, so anything sleep did NOT consolidate can be written again from
    # its own transcript under the new weights — no original text needed. This
    # measures that path: replay the decoded transcript back into the store,
    # segment by segment, then read it.
    model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
    S_ = args.seg_len
    # r32 fix: replay each segment's ORIGINAL selection instead of recomputing it.
    # A content-based write policy is a function of the weights, and sleep moved
    # them, so re-scoring here builds a DIFFERENT store — measured at 6.3% against
    # 100% for a fixed policy. `slot_pos` recorded the source position of each slot.
    # The transcript is the DECODED STORE -- `decode_store` returns one token per
    # slot, in slot order. So it is already the selection: slot j is transcript
    # position j, one for one. Re-encoding it therefore writes EVERY transcript
    # position rather than re-running a selection policy over it.
    #
    # This is the correction to the r32/r33 framing. `slot_pos` records positions
    # in the original 128-token wake segment, while the transcript is 96 tokens in
    # slot coordinates -- forcing one onto the other mixes two coordinate systems.
    # Re-scoring (r32) is worse still: after consolidation the facts are the
    # tokens the model predicts BEST, so a surprisal gate ranks them lowest and
    # drops exactly the content worth keeping. That is r6's inversion again.
    #
    # A full-rate rewrite has no policy to be unstable, which is why r28 measured
    # 100% here without needing any of this.
    reencode_all = not getattr(args, "reencode_rescore", False)
    mem_cfg_saved, wh_cfg_saved = model.memory.cfg, model.memory.write_head.cfg
    if reencode_all:
        # WriteHead keeps its own reference, so both have to be overridden.
        flat = replace(mem_cfg_saved, write_select="all", write_frac=1.0)
        model.memory.cfg = flat
        model.memory.write_head.cfg = flat
    chunks_written = 0
    with torch.no_grad():
        for i in range(0, replay.shape[1], S_):
            chunk = replay[:, i:i + S_]
            if chunk.shape[1] < 2:
                continue
            chunks_written += 1
            _, h_w = model(chunk)
            model.memory.commit(h_w, score=getattr(model, "write_score", None))
    model.memory.cfg, model.memory.write_head.cfg = mem_cfg_saved, wh_cfg_saved
    h, t = answer(model, *probe_q, read=True)
    reencoded_store = h / t

    drift = float(sum(((pre[k] - v).norm() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point and pre[k].shape == v.shape)))

    out = {
        "slots": n_slots,
        # How much of the re-encode actually replayed the original selection.
        # If `unforced` is nonzero those chunks were RE-SCORED against the slept
        # weights, which is precisely the failure r32 measured — so the number is
        # reported rather than left to be assumed zero.
        # How the transcript was re-encoded. "all" is a 1:1 slot rewrite, which
        # has no content-dependent policy to be unstable across sleep.
        "reencode_policy": "all" if reencode_all else "rescored",
        "reencode_chunks": chunks_written,
        "decode_exact": decode_exact,
        "wake_recall_with_store": wake_recall,
        "wake_recall_heldout_fact": wake_recall_heldout,
        "weights_alone_before_sleep": before_weights,
        "weights_alone_after_sleep": after_weights,
        "weights_alone_after_sleep_replayed_fact": after_weights_replayed,
        "slots_reclaimable": int(round(after_weights * n_slots)),
        "post_sleep_store_recall": post_sleep_store,
        "store_rewritten_after_sleep": rewritten_store,
        "store_reencoded_from_transcript": reencoded_store,
        "lm_loss_before": lm_before,
        "lm_loss_after": lm_after,
        "weight_drift_l2": drift,
        "sleep_steps": a.sleep_steps,
        "sleep_lr": a.sleep_lr,
        "freeze_keys": a.freeze_keys,
        "rehearse": a.rehearse,
        "rehearse_mem": a.rehearse_mem,
        "mem_batch": a.mem_batch,
        "tag": a.tag,
        "shared_facts": a.shared_facts,
        "n": t,
    }
    print(json.dumps(out, indent=2))
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
