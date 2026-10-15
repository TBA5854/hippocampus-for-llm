"""
sleep_cycles.py — the eviction loop in steady state.

r27/r28 measured ONE wake -> sleep. That is the easy case: the weights start
where training left them, sleep moves them once, and nothing has to survive
afterwards. A memory that runs for a day does this over and over, and the
questions that decides are different:

  RETENTION    do facts consolidated in cycle 1 survive cycles 2..N, or does
               each sleep overwrite the last one's? (catastrophic forgetting
               between users, which single-cycle runs cannot see)
  DRIFT        r27 found a cliff: at weight drift ~418 the store becomes
               unreadable. Does drift accumulate across cycles toward it?
  THROUGHPUT   slots reclaimed per cycle vs slots filled per wake. If sleep
               cannot keep up, the store overflows regardless.
  LIVELOCK     facts that never consolidate pin their slots forever.

Each cycle is one "user": a fresh fact set, written to a store that starts
empty (the previous cycle's slots having been freed by consolidation), then
consolidated and verified. Verification for EVERY user seen so far runs with
memory off and nothing in context, so a correct answer can only come from
weights.

    uv run --group sleep python -m mech.sleep_cycles --run R.json --ckpt C.pt --cycles 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mech.persist import restore
from mech.sleep_evict import (answer, decode_store, held_out_query, lm_loss,
                              share_facts, sleep, synth_queries, wake)
from mech.tasks import _sym
from mech.train import get_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mem-batch", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=6)
    ap.add_argument("--sleep-steps", type=int, default=1500)
    ap.add_argument("--sleep-lr", type=float, default=3e-4)
    ap.add_argument("--rehearse", type=float, default=1.0)
    ap.add_argument("--rehearse-mem", type=float, default=3.0)
    ap.add_argument("--replay-past", type=float, default=0.0,
                    help="also replay EARLIER users' transcripts each sleep, at this\n"
                         "                        weight — the obvious guard if retention fails")
    ap.add_argument("--replay-cap", type=int, default=512,
                    help="max supervised rows per sleep when replaying the past.\n"
                         "                        Replaying every user in full grows linearly and OOMs by ~user 5;\n"
                         "                        a FIXED budget is also the realistic design — a night is finite")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    qtok = _sym(tok.vocab_size, getattr(args, "n_symbols", 32))[2]
    gen = torch.Generator().manual_seed(1234)
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}

    users, cycles = [], []
    for c in range(a.cycles):
        # ---- WAKE: a new user's facts into a store that starts empty --------
        x, y, m = get_batch(val_ids, tok, args, device, gen)
        x, y, m = share_facts(x, y, m, args.n_facts, qtok)   # one user per cycle
        probe_q = held_out_query(x, args.n_facts, gen)
        wake(model, x, device)
        n_slots = int(model.memory.store.n_stored[0])
        replay = decode_store(model, n_slots)
        h, t = answer(model, *probe_q, read=True)
        wake_recall = h / t
        users.append({"probe": probe_q, "replay": replay, "x": x})

        # ---- SLEEP on this user's transcript --------------------------------
        supervised = synth_queries(replay, args.n_facts, val_ids, tok, args,
                                   device, gen, qtok)
        if a.replay_past > 0 and len(users) > 1:
            # concatenate earlier users' supervised queries into this sleep
            xs, ys, ms = [supervised[0]], [supervised[1]], [supervised[2]]
            # the current user keeps its full set; the past shares what is left of
            # the budget, so sleep cost per night is constant as users accumulate
            past_budget = max(a.replay_cap - supervised[0].shape[0], 0)
            per_user = max(past_budget // max(len(users) - 1, 1), 1)
            for u in users[:-1]:
                s2 = synth_queries(u["replay"], args.n_facts, val_ids, tok, args,
                                   device, gen, qtok)
                if s2[0].shape[0] > per_user:
                    idx = torch.randperm(s2[0].shape[0], generator=gen)[:per_user].to(device)
                    s2 = (s2[0][idx], s2[1][idx], s2[2][idx])
                xs.append(s2[0]); ys.append(s2[1]); ms.append(s2[2])
            supervised = (torch.cat(xs), torch.cat(ys), torch.cat(ms))
        sleep(model, replay, supervised, a.sleep_steps, a.sleep_lr,
              args.answer_weight, device, rehearse=a.rehearse,
              rehearse_mem=a.rehearse_mem, ids=val_ids, args=args, tok=tok,
              mem_batch=a.mem_batch)

        # ---- VERIFY: every user so far, from weights alone -------------------
        retention = []
        for u in users:
            h, t = answer(model, *u["probe"], read=False)
            retention.append(round(h / t, 4))

        # ---- can the store still be written and read after this sleep? -------
        wake(model, x, device)
        h, t = answer(model, *probe_q, read=True)
        rewritten = h / t

        drift = float(sum(((start[k] - v).norm()
                           for k, v in model.state_dict().items()
                           if v.dtype.is_floating_point and start[k].shape == v.shape)))
        cycles.append({
            "cycle": c,
            "slots_filled": n_slots,
            "wake_recall": wake_recall,
            "consolidated_this_user": retention[-1],
            "retention_all_users": retention,
            "retention_oldest": retention[0],
            "store_rewritten_after_sleep": rewritten,
            "lm_loss": lm_loss(model, val_ids, args, device),
            "cumulative_drift_l2": drift,
        })
        print(json.dumps(cycles[-1]))

    out = {"tag": a.tag, "cycles": cycles, "n_users": a.cycles,
           "sleep_steps": a.sleep_steps, "sleep_lr": a.sleep_lr,
           "rehearse": a.rehearse, "rehearse_mem": a.rehearse_mem,
           "replay_past": a.replay_past, "replay_cap": a.replay_cap,
           "batch": a.batch}
    print(json.dumps(out, indent=2))
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
