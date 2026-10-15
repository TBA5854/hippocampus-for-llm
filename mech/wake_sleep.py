"""
wake_sleep.py — the ever-learning loop on the trained layer.

Three phases, and they are not the same thing:

  PRETRAIN   many RANDOM personas, HC on, everything trainable.
             Teaches the mechanism (how to use a store), not any facts.
  WAKE       one held-out persona. Weights FROZEN, model in eval, HC on.
             All adaptation lives in the store.
  SLEEP      HC READ DISABLED, weights unfrozen, replay the persona's
             query->answer pairs. Reads are off on purpose: with retrieval live
             the model satisfies the target by looking it up, the weights never
             absorb anything, and the memory becomes a crutch that hides its own
             failure to transfer.

Then probe with HC off and nothing in context. A correct answer there can only
have come from the weights, which is the definition of consolidation.

    uv run --group sleep python -m mech.wake_sleep --pretrain-steps 4000

Honest limitation: replay is reconstructed from the persona, not decoded from
the store. The store holds pooled d_mem summaries and no token ids, so true
replay-from-store needs the parallel token buffer v0.1 had. Noted rather than
papered over — it means this measures "can sleep write facts into weights",
not yet "can the store itself drive replay".
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

from mech.config import MemConfig
from mech.data import load
from mech.hippo_layer import HippoMemory
from mech.model import GPTConfig, TinyGPT
from mech.tasks import TASK_CHARS, Persona, exact_match, persona_batch


def run(model, x, y, m, answer_weight, device, train: bool):
    B, n_seg, _ = x.shape
    if model.memory is not None:
        model.memory.reset(B, device=device, dtype=model.tok_emb.weight.dtype)
    loss_sum, hits, tot = None, 0, 0
    for s in range(n_seg):
        logits, h_write = model(x[:, s])
        flat = logits.reshape(-1, logits.size(-1))
        loss = F.cross_entropy(flat, y[:, s].reshape(-1))
        if m[:, s].any():
            sel = m[:, s].reshape(-1)
            loss = loss + answer_weight * F.cross_entropy(
                flat[sel], y[:, s].reshape(-1)[sel])
            h, t = exact_match(logits.detach(), y[:, s], m[:, s])
            hits += h; tot += t
        loss_sum = loss if loss_sum is None else loss_sum + loss
        if model.memory is not None and h_write is not None:
            model.memory.commit(h_write.detach())
    return loss_sum / n_seg, hits, tot


@torch.no_grad()
def probe(model, ids, tok, persona, args, device, *, hc_read: bool,
          no_facts: bool, facts_in_context: bool = False, iters: int = 10):
    was_train, mem = model.training, model.memory
    model.eval()
    if mem is not None:
        prev = mem.read_enabled
        mem.set_read(hc_read)
    g = torch.Generator().manual_seed(999)
    hits = tot = 0
    for _ in range(iters):
        x, y, m = persona_batch(ids, tok, persona, args.batch, args.n_seg,
                                args.seg_len, device, g,
                                facts_in_context=facts_in_context,
                                no_facts=no_facts)
        _, h, t = run(model, x, y, m, args.answer_weight, device, train=False)
        hits += h; tot += t
    if mem is not None:
        mem.set_read(prev)
    if was_train:
        model.train()
    return hits / tot if tot else 0.0


@torch.no_grad()
def lm_loss(model, ids, tok, args, device, iters=10):
    """Held-out LM loss — catches consolidation damaging general ability."""
    from mech.data import sample
    was_train = model.training
    model.eval()
    g = torch.Generator().manual_seed(4242)
    tot = 0.0
    for _ in range(iters):
        x, y = sample(ids, args.batch, args.n_seg, args.seg_len, device, g)
        m = torch.zeros_like(x, dtype=torch.bool)
        l, _, _ = run(model, x, y, m, 0.0, device, train=False)
        tot += float(l)
    if was_train:
        model.train()
    return tot / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--sleep-steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-seg", type=int, default=4)
    ap.add_argument("--seg-len", type=int, default=128)
    ap.add_argument("--n-facts", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=65)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--d-mem", type=int, default=128)
    ap.add_argument("--memory-size", type=int, default=64)
    ap.add_argument("--answer-weight", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sleep-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ceiling", type=float, default=0.5,
                    help="warn if the copy ceiling is below this")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, train_ids, val_ids = load(args.data, extra_chars=TASK_CHARS)

    mcfg = MemConfig(d_model=args.d_model, d_mem=args.d_mem,
                     memory_size=args.memory_size, read_layers=(1,),
                     write_layer=4)
    model = TinyGPT(GPTConfig(vocab_size=tok.vocab_size, d_model=args.d_model,
                              n_layers=args.n_layers, n_heads=args.n_heads,
                              seg_len=args.seg_len), HippoMemory(mcfg)).to(device)
    print(f"params={model.n_params()/1e6:.2f}M  device={device}  "
          f"facts/persona={args.n_facts}")

    # ---- PRETRAIN: random personas, so the mechanism generalises -----------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.pretrain_steps, pct_start=0.1)
    g = torch.Generator().manual_seed(args.seed)
    print("\n== PRETRAIN (random personas, HC on, all trainable) ==")
    for step in range(1, args.pretrain_steps + 1):
        p = Persona(args.n_facts, g)
        x, y, m = persona_batch(train_ids, tok, p, args.batch, args.n_seg,
                                args.seg_len, device, g)
        loss, h, t = run(model, x, y, m, args.answer_weight, device, train=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % max(args.pretrain_steps // 8, 1) == 0:
            print(f"  step {step:>5}  loss {float(loss.detach()):.4f}  "
                  f"train-acc {100*h/max(t,1):5.1f}%  "
                  f"gates={model.memory.gate_magnitudes()}")

    # ---- held-out persona --------------------------------------------------
    hp = Persona(args.n_facts, torch.Generator().manual_seed(12345))
    print(f"\nheld-out persona: {hp.facts}")

    def table(tag):
        return {
            "HC on, facts in seg 0": probe(model, val_ids, tok, hp, args, device,
                                           hc_read=True, no_facts=False),
            "HC off, facts in seg 0": probe(model, val_ids, tok, hp, args, device,
                                            hc_read=False, no_facts=False),
            "HC off, nothing (weights only)": probe(model, val_ids, tok, hp, args,
                                                    device, hc_read=False,
                                                    no_facts=True),
            "ceiling: facts in query segment": probe(model, val_ids, tok, hp, args,
                                                     device, hc_read=False,
                                                     no_facts=True,
                                                     facts_in_context=True),
            "LM val loss": lm_loss(model, val_ids, tok, args, device),
        }

    before = table("before")
    print("\n== WAKE (weights frozen) ==")
    for k, v in before.items():
        print(f"  {k:34s} {v:.4f}" if "LM" in k else f"  {k:34s} {100*v:5.1f}%")

    # GATE. The ceiling puts the facts in the same segment as the query, so it is
    # a pure copy. If the model cannot do that, it never acquired the mechanism,
    # the wake row measures nothing, and any post-sleep number is just rote
    # memorisation of a handful of pairs — which looks like a triumph (every
    # condition hits 100%, including the one with no facts and no retrieval) and
    # means nothing. Measured at d_model=65: ceiling 0.0%.
    ceiling = before["ceiling: facts in query segment"]
    if ceiling < args.min_ceiling:
        print(f"\n  *** CEILING {100*ceiling:.1f}% < {100*args.min_ceiling:.0f}% — "
              f"the mechanism was not learned. Everything below is\n"
              f"      unfalsifiable: sleep will still reach 100% by memorising "
              f"{args.n_facts} pairs.\n"
              f"      Fix pretraining (width / steps / task difficulty) before "
              f"reading these numbers. ***")

    # ---- SLEEP: reads OFF, weights move ------------------------------------
    print(f"\n== SLEEP ({args.sleep_steps} steps, HC READ DISABLED) ==")
    model.memory.set_read(False)
    model.train()
    sopt = torch.optim.AdamW(model.parameters(), lr=args.sleep_lr)
    sg = torch.Generator().manual_seed(777)
    for step in range(1, args.sleep_steps + 1):
        # replay: the persona's query->answer pairs, nothing in context
        x, y, m = persona_batch(train_ids, tok, hp, args.batch, args.n_seg,
                                args.seg_len, device, sg, no_facts=True)
        loss, h, t = run(model, x, y, m, args.answer_weight, device, train=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sopt.step(); sopt.zero_grad(set_to_none=True)
        if step % max(args.sleep_steps // 4, 1) == 0:
            print(f"  step {step:>4}  loss {float(loss.detach()):.4f}  "
                  f"replay-acc {100*h/max(t,1):5.1f}%")
    model.memory.set_read(True)

    after = table("after")
    print("\n== AFTER SLEEP ==")
    print(f"  {'condition':34s} {'before':>9}  {'after':>9}")
    for k in before:
        if "LM" in k:
            print(f"  {k:34s} {before[k]:9.4f}  {after[k]:9.4f}")
        else:
            print(f"  {k:34s} {100*before[k]:8.1f}%  {100*after[k]:8.1f}%")


if __name__ == "__main__":
    main()
