"""Demo: the shared store, its exact-match read, and what makes it a memory.

Runs on a CPU in under a minute and needs no data and no GPU.

What it shows (the store mechanics, with a hand-built embedding table and no trained model):
  1. Facts are written as (key = the token's position-free embedding, value = what followed it).
  2. A query is the token's own embedding, so the read is an exact match: recall is perfect.
  3. The read is sparse (top-1) and stays exact as the store grows to 100,000 slots.
  4. Controls: reading a scrambled store, or an empty store, gives chance.
  5. The store is a file: saved, reloaded in a fresh object, it answers the same;
     a file stamped by another model is refused.

What it does NOT show: a trained language model learning to use the store. That needs GPU runs;
the numbers are in experiment/ and the journal manuscript.

    python demo/store_demo.py
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run from anywhere

import torch
import torch.nn.functional as F

from mech.config import MemConfig
from mech.hippo_layer import EpisodicStore, settle

torch.manual_seed(0)
VOCAB, D, N_FACTS = 8192, 256, 64
EMB = F.normalize(torch.randn(VOCAB, D), dim=-1)          # stand-in for a model's token embeddings


def make_store(n_slots, grow=False):
    cfg = MemConfig(d_model=D, d_mem=D, memory_size=n_slots, readout="topk", top_k=1,
                    beta=32.0, store_grow=grow)
    store = EpisodicStore(cfg)
    store.reset(1)
    store.eval()                                          # eval mode: in-place O(S) writes
    return cfg, store


def write_facts(store, keys, vals):
    store.write_many(EMB[keys][None], EMB[vals][None], tok=keys[None])


def read(cfg, store, query_tokens):
    r, _ = settle(EMB[query_tokens][None], store.keys, store.values, store.valid_mask(), cfg)
    return (r[0] @ EMB.T).argmax(-1)                      # nearest embedding = the recalled token


def recall(cfg, store, keys, vals):
    return (read(cfg, store, keys) == vals).float().mean().item() * 100


def main():
    t0 = time.time()
    g = torch.Generator().manual_seed(1)
    keys = torch.randperm(VOCAB, generator=g)[:N_FACTS]
    vals = torch.randint(0, VOCAB, (N_FACTS,), generator=g)

    cfg, store = make_store(512)
    write_facts(store, keys, vals)
    print(f"1-2. {N_FACTS} facts written once, queried by their own key tokens")
    print(f"     exact-match recall: {recall(cfg, store, keys, vals):.1f}%   "
          f"(slots used: {int(store.n_stored[0])})")

    # 3. grow to 100,000 distractor slots; the read still touches one slot and stays exact
    cfg, store = make_store(1024, grow=True)
    write_facts(store, keys, vals)
    n_extra = 100_000
    pool = torch.tensor(sorted(set(range(VOCAB)) - set(keys.tolist())))   # tokens that are not fact keys
    junk_k = pool[torch.randint(0, len(pool), (n_extra,), generator=g)]
    junk_v = torch.randint(0, VOCAB, (n_extra,), generator=g)
    SEG = 128                                             # the model writes one 128-token segment at a time
    for i in range(0, n_extra, SEG):
        write_facts(store, junk_k[i:i + SEG], junk_v[i:i + SEG])
    n_slots = int(store.n_stored[0])
    ms = time.time()
    rec = recall(cfg, store, keys, vals)
    print(f"3.   store grown to {n_slots:,} slots of other tokens; top-1 read (1 slot used per query)")
    print(f"     recall: {rec:.1f}%  [{time.time() - ms:.2f}s for {N_FACTS} queries]")

    # 4. controls: a scrambled store (another example's store) and an empty store
    cfg, store = make_store(512)
    write_facts(store, keys, vals)
    other = torch.randperm(VOCAB, generator=g)[:N_FACTS]
    cfg_o, store_o = make_store(512)
    write_facts(store_o, other, torch.randint(0, VOCAB, (N_FACTS,), generator=g))
    print("4.   controls")
    print(f"     scrambled store (another store read with these queries): "
          f"{recall(cfg, store_o, keys, vals):.1f}%")
    cfg_e, store_e = make_store(512)
    print(f"     empty store: {recall(cfg_e, store_e, keys, vals):.1f}%")

    # 5. the store is a file
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "store.pt"
        store.save(path, fingerprint="model-A")
        cfg2, fresh = make_store(512)
        fresh.load(path, fingerprint="model-A")
        print(f"5.   saved, reloaded into a fresh store: recall {recall(cfg2, fresh, keys, vals):.1f}%")
        try:
            fresh.load(path, fingerprint="model-B")
            print("     foreign model's file: ACCEPTED (unexpected)")
        except ValueError as e:
            print(f"     foreign model's file refused: {str(e)[:60]}...")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
