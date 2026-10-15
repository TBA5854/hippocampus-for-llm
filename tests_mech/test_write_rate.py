"""
Tests for write-rate selection (r30).

One slot per token is redundant: with causal attention, position t's state
already summarises everything up to t, so an episode of T tokens is stored T
times over. These select which positions earn a slot. What must hold:

  * the default ("all") is byte-identical to the old behaviour — every earlier
    round depends on it
  * each mode stores the number of slots it claims to
  * "surprisal" keeps the positions with the highest score, which is what makes
    it different from stride: it is content-aware

    uv run --group sleep pytest tests_mech/ -q
"""

from dataclasses import replace

import pytest
import torch

from mech.config import MemConfig
from mech.hippo_layer import HippoMemory, WriteHead

CFG = MemConfig(d_model=16, d_mem=16, memory_size=256, read_layers=(1,),
                write_layer=3, write_per_token=True, value_shift=1)


def _commit(cfg, S=32, B=2, score=None):
    m = HippoMemory(cfg)
    m.reset(B)
    h = torch.randn(B, S, cfg.d_model)
    m.commit(h, score=score)
    return m, h


def test_default_writes_every_position():
    m, _ = _commit(CFG, S=32)
    assert int(m.store.n_stored[0]) == 31          # S - value_shift


def test_stride_writes_every_kth():
    for k, want in ((2, 16), (4, 8)):
        m, _ = _commit(replace(CFG, write_select="stride", write_stride=k), S=32)
        assert int(m.store.n_stored[0]) == want


def test_last_writes_one_slot():
    m, _ = _commit(replace(CFG, write_select="last"), S=32)
    assert int(m.store.n_stored[0]) == 1


def test_surprisal_keeps_the_fraction_it_claims():
    S = 32
    score = torch.rand(2, S - 1)
    m, _ = _commit(replace(CFG, write_select="surprisal", write_frac=0.25),
                   S=S, score=score)
    assert int(m.store.n_stored[0]) == round(0.25 * (S - 1))


def test_surprisal_keeps_the_most_surprising_positions():
    # a score with an obvious top-4, and stride would miss three of them
    S = 17
    score = torch.zeros(1, S - 1)
    score[0, [1, 2, 3, 4]] = 10.0
    hk = torch.arange(S - 1, dtype=torch.float32)[None, :, None].repeat(1, 1, 4)
    hv = hk.clone()
    cfg = replace(CFG, write_select="surprisal", write_frac=4 / (S - 1))
    kept_k, _, kept_pos = WriteHead._select(hk, hv, cfg, score)
    kept = sorted(int(v) for v in kept_k[0, :, 0])
    assert kept == [1, 2, 3, 4]
    assert sorted(int(v) for v in kept_pos[0]) == [1, 2, 3, 4]


def test_no_score_falls_back_to_the_same_rate():
    # surprisal without a score must not silently write everything
    m, _ = _commit(replace(CFG, write_select="surprisal", write_frac=0.5), S=32)
    assert int(m.store.n_stored[0]) <= 16


def test_key_proj_query_matches_the_write_projection():
    """Narrow stores cannot use an identity query; key_proj is its analogue.

    r9's requirement is that query and key are comparable. With
    write_identity=False the key is `to_key(h)` for a frozen random to_key, so
    the query must be that same map — not I, which does not even have the right
    shape when d_mem != d_model.
    """
    cfg = replace(CFG, d_mem=8, d_model=16, write_identity=False,
                  query_init="key_proj", freeze_query=True)
    m = HippoMemory(cfg)
    head = next(iter(m.read_heads.values()))
    assert torch.equal(head.dg.proj.weight, m.write_head.to_key.weight)
    assert not head.dg.proj.weight.requires_grad


def test_slots_remember_the_position_they_came_from():
    """r32: the surprisal gate is a function of the weights, so after sleep the
    same text scores differently and a recomputed selection is a DIFFERENT store.
    Slots therefore record their source position so an episode can be rewritten
    exactly as it was."""
    S = 32
    score = torch.rand(2, S - 1)
    cfg = replace(CFG, write_select="surprisal", write_frac=0.25)
    m, _ = _commit(cfg, S=S, score=score)
    n = int(m.store.n_stored[0])
    pos = m.store.slot_pos[0, :n]
    assert int((pos >= 0).sum()) == n
    assert pos.tolist() == sorted(score[0].topk(n).indices.tolist())


def test_a_forced_position_past_the_segment_is_a_clear_error():
    """r33 crashed here. A forced position indexes the segment it was WRITTEN
    over, and the re-encode replays a chunk that can be shorter (the final
    remainder), so the position can land past the end. As a bare gather that is
    a CUDA device-side assert, reported asynchronously at the next sync point —
    the traceback pointed at an unrelated line in write_many and cost a run."""
    S = 8
    hk = torch.randn(2, S, 4)
    cfg = replace(CFG, write_select="surprisal", write_frac=0.5)
    bad = torch.tensor([[0, 99], [1, 2]])
    with pytest.raises(IndexError, match="forced write positions"):
        WriteHead._select(hk, hk.clone(), cfg, None, force=bad)


def test_a_forced_selection_reproduces_an_earlier_one():
    S, B = 32, 2
    cfg = replace(CFG, write_select="surprisal", write_frac=0.25)
    first = _commit(cfg, S=S, score=torch.rand(B, S - 1))[0]
    n = int(first.store.n_stored[0])
    want = first.store.slot_pos[:, :n]

    # a DIFFERENT score: without forcing, the selection would change
    m2 = HippoMemory(cfg)
    m2.reset(B)
    h = torch.randn(B, S, cfg.d_model)
    m2.commit(h, score=torch.rand(B, S - 1), force_idx=want)
    got = m2.store.slot_pos[:, :n]
    assert torch.equal(got, want)
