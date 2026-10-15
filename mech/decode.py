"""
decode.py — can a stored VALUE be read back as a token id?

This is the prerequisite for sleep-as-eviction. `wake_sleep.py` consolidates
facts into weights by replaying them, but its replay is reconstructed from the
persona it already knows, not decoded from the store — its own docstring flags
that as the honest limitation, because the v0.1-v0.3 store held pooled d_mem
summaries with no token identity in them.

The current store is different: writes are per token, and the value carries the
OBSERVED token (`--value-shift 1`) with a skip from block 0, which is where
token identity still lives (r9: a block-0 value holds facts seen once at 100%,
a last-block value does not). So the value should be decodable. "Should" is
why this file exists.

Three decoders, cheapest first:

  emb    nearest neighbour against the token embedding matrix, cosine
  head   the model's own LM head applied to the stored vector
  probe  a linear map fitted on (stored value -> token id) from a held-out
         batch; an upper bound on what a *learned* decoder could recover

A high `emb`/`head` number means replay can be driven by the store with no new
parameters. If only `probe` is high, replay needs a small trained decoder — a
worse but workable answer. If none is high, the value does not carry token
identity and sleep cannot free slots on its own.

    uv run --group sleep python -m mech.decode --run RESULT.json --ckpt CKPT.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from mech.persist import restore
from mech.train import get_batch


@torch.no_grad()
def fill(model, val_ids, tok, args, device, gen):
    """Write one batch's fact + distractor segments into the store.

    Returns the store's values and, for each slot, the token that was observed
    at the position it was written from — the ground truth to decode back to.
    """
    x, _, _ = get_batch(val_ids, tok, args, device, gen)
    n_seg = x.shape[1]
    model.memory.reset(x.shape[0], device=device, dtype=model.tok_emb.weight.dtype)
    for s in range(n_seg - 1):
        _, h_write = model(x[:, s])
        # the score must be passed, or write_select="surprisal" silently falls
        # back to a stride of the same rate and the probe measures a store the
        # training run never built
        model.memory.commit(h_write, score=getattr(model, "write_score", None))

    store = model.memory.store
    shift = model.memory.cfg.value_shift
    S = x.shape[2]
    per = S - shift
    # slot i of segment s was written from position (i + shift) of that segment,
    # which is exactly what `commit` pairs with the key at position i
    truth = torch.cat([x[:, s, shift:] for s in range(n_seg - 1)], dim=1)
    # the KEY of slot i is the token at position i — with position-free layer-0
    # keys that is the token embedding itself, so keys should decode exactly
    key_truth = torch.cat([x[:, s, :S - shift] for s in range(n_seg - 1)], dim=1)
    n = min(int(store.n_stored[0]), truth.shape[1])
    return store.values[:, :n], truth[:, :n], per, store.keys[:, :n], key_truth[:, :n]


def decode_emb(values, emb):
    v = F.normalize(values, dim=-1)
    e = F.normalize(emb, dim=-1)
    return (v @ e.T).argmax(-1)


def decode_head(model, values):
    return model.head(model.ln_f(values)).argmax(-1)


def fit_probe(values, truth, vocab, steps=300, lr=1e-2):
    """Linear (value -> token) fitted on this batch; reported on a held-out one."""
    d = values.shape[-1]
    W = torch.zeros(d, vocab, device=values.device, requires_grad=True)
    opt = torch.optim.Adam([W], lr=lr)
    flat_v, flat_t = values.reshape(-1, d), truth.reshape(-1)
    for _ in range(steps):
        loss = F.cross_entropy(flat_v @ W, flat_t)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return W.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="train.py --out JSON")
    ap.add_argument("--ckpt", type=Path, required=True, help="train.py --save state_dict")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--probe-steps", type=int, default=300)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args, tok, val_ids, model = restore(a.run, a.ckpt, a.batch, device)
    gen = torch.Generator().manual_seed(1234)

    values, truth, per, keys, key_truth = fill(model, val_ids, tok, args, device, gen)
    emb = model.tok_emb.weight

    acc = {}
    # emb and head both read the value as if it lived in d_model space, which it
    # does only when the write is the identity (d_mem == d_model). Otherwise the
    # learned probe is the only decoder that applies.
    same_space = values.shape[-1] == emb.shape[-1]
    if same_space:
        acc["emb"] = float((decode_emb(values, emb) == truth).float().mean())
        acc["head"] = float((decode_head(model, values) == truth).float().mean())
    else:
        acc["emb"] = acc["head"] = None

    # probe: fit on this batch, report on a fresh one (no train-on-test)
    W = fit_probe(values, truth, emb.shape[0], steps=a.probe_steps)
    v2, t2, _, k2, kt2 = fill(model, val_ids, tok, args, device, gen)
    acc["probe_heldout"] = float(((v2 @ W).argmax(-1) == t2).float().mean())

    # How much of the store is fact symbols vs distractor text. Most slots hold
    # text, so an overall number is dominated by it: a decoder that never gets a
    # symbol right can still score ~95% and be useless for replay. Every decoder
    # is therefore split, the PROBE included — that split is the number that
    # decides whether sleep can replay from the store.
    nsym = getattr(args, "n_symbols", 32)
    kbase = tok.vocab_size
    is_sym = truth >= kbase
    sym2 = t2 >= kbase
    pairs = [("probe", (v2 @ W).argmax(-1), t2, sym2)]
    if same_space:
        pairs += [("emb", decode_emb(values, emb), truth, is_sym),
                  ("head", decode_head(model, values), truth, is_sym)]
    for name, pred, tgt, mask in pairs:
        if bool(mask.any()):
            acc[f"{name}_on_symbols"] = float((pred[mask] == tgt[mask]).float().mean())
        acc[f"{name}_on_text"] = float((pred[~mask] == tgt[~mask]).float().mean())

    # KEYS. With position-free layer-0 keys the key IS the token embedding, so
    # nearest neighbour should be near-exact. That matters more than the value:
    # slot i+1's key is slot i's value token, so the key sequence alone
    # reconstructs the written text — replay without decoding values at all.
    # keys are stored raw; centring (`K - key_mean`) happens at read time, so
    # what is on disk is directly comparable to the embedding matrix
    kpred = decode_emb(keys, emb)
    acc["key_emb"] = float((kpred == key_truth).float().mean())
    ksym = key_truth >= tok.vocab_size
    if bool(ksym.any()):
        acc["key_emb_on_symbols"] = float((kpred[ksym] == key_truth[ksym]).float().mean())
    acc["key_emb_on_text"] = float((kpred[~ksym] == key_truth[~ksym]).float().mean())
    # the value token of slot i read off slot i+1's key
    acc["value_via_next_key"] = float((kpred[:, 1:] == truth[:, :-1]).float().mean())
    vsym = truth[:, :-1] >= tok.vocab_size
    if bool(vsym.any()):
        acc["value_via_next_key_on_symbols"] = float(
            (kpred[:, 1:][vsym] == truth[:, :-1][vsym]).float().mean())

    # COVERAGE (r30 H30d). Write-rate selection makes the store a SUBSAMPLED
    # transcript, so the question is no longer "does a slot decode" but "is the
    # fact in the store at all". Recall can survive holes; replay cannot, if the
    # holes fall on fact tokens. Measured as: of the fact KEY tokens written in
    # segment 0, what fraction appear among the decoded keys of that example.
    n_facts = int(getattr(args, "n_facts", 8))
    x_cov, _, _ = get_batch(val_ids, tok, args, device,
                            torch.Generator().manual_seed(1234))
    fact_keys = x_cov[:, 0, :2 * n_facts:2]                  # [B, n_facts]
    covered = 0
    for b in range(kpred.shape[0]):
        present = set(int(v) for v in kpred[b].tolist())
        covered += sum(1 for k in fact_keys[b].tolist() if int(k) in present)
    acc["fact_key_coverage"] = covered / (kpred.shape[0] * n_facts)

    sel = getattr(args, "write_select", "all")
    out = {"slots": int(values.shape[1]), "batch": a.batch,
           "symbol_fraction": float(is_sym.float().mean()),
           "n_symbols": nsym, "write_select": sel, **acc}
    if sel != "all":
        # every per-slot accuracy above pairs slot i with position i, which only
        # holds when every position is written. Under selection that mapping is
        # gone, so those numbers are meaningless — coverage is not, because it
        # asks which tokens are PRESENT, not which slot holds them.
        out["alignment_metrics_valid"] = False
        for k in ("emb", "head", "probe_heldout", "key_emb", "value_via_next_key"):
            out.pop(k, None)
            out.pop(f"{k}_on_symbols", None)
            out.pop(f"{k}_on_text", None)
    print(json.dumps(out, indent=2))
    if a.out:
        a.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
