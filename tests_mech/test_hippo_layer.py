"""
Tests for the trained episodic-memory layer.

Deliberately NOT under tests/ — that suite imports TensorFlow, and TF and torch
cannot share a process on this hardware (segfault at import; see
harness/memory.py). Keeping them in separate directories keeps them in separate
pytest processes.

    uv run --group sleep pytest tests_mech/ -q
"""

from dataclasses import replace

import pytest
import torch

from mech.config import MemConfig
from mech.hippo_layer import EpisodicStore, HippoMemory, ReadHead, WriteHead, settle

torch.manual_seed(0)
CFG = MemConfig(d_model=32, d_mem=32, memory_size=8, read_layers=(1,),
                write_layer=3, write_per_token=False)
PER_TOKEN = replace(CFG, memory_size=64, write_per_token=True)


def _mem(cfg=CFG, batch=2):
    m = HippoMemory(cfg)
    m.reset(batch)
    return m


# ------------------------------------------------------------ retrieval ----
@pytest.mark.parametrize("readout", ["graded", "ste", "topk"])
def test_retrieves_the_matching_slot(readout):
    """All three readouts must return the value whose key the query matches."""
    cfg = replace(CFG, readout=readout, beta=16.0)
    B, M, D = 1, 4, 8
    keys = torch.eye(M, D)[None]                       # orthogonal keys
    values = torch.arange(M * D, dtype=torch.float32).reshape(1, M, D)
    mask = torch.ones(1, M, dtype=torch.bool)
    for slot in range(M):
        q = keys[:, slot][:, None, :]                  # [1, 1, D] — exactly key `slot`
        r, _ = settle(q, keys, values, mask, cfg)
        best = torch.argmin(((values[0] - r[0, 0]) ** 2).sum(-1))
        assert best.item() == slot, f"{readout} retrieved slot {best} for query {slot}"


def test_empty_store_returns_exact_zeros_not_nan():
    """0 * NaN is NaN, which would break the zero-init no-op below."""
    cfg = CFG
    keys = torch.zeros(2, cfg.memory_size, cfg.d_mem)
    mask = torch.zeros(2, cfg.memory_size, dtype=torch.bool)
    r, sim = settle(torch.randn(2, 3, cfg.d_mem), keys, keys, mask, cfg)
    assert torch.isfinite(r).all() and torch.isfinite(sim).all()
    assert torch.count_nonzero(r) == 0


# ---------------------------------------------------------------- store ----
def test_store_persists_across_segments_and_resets_between_examples():
    m = _mem()
    assert m.store.n_stored.tolist() == [0, 0]
    for i in range(3):
        m.commit(torch.randn(2, 5, CFG.d_model) * (i + 1))
    assert (m.store.n_stored > 0).all(), "nothing was written across three segments"
    before = m.store.n_stored.clone()
    m.reset(2)
    assert m.store.n_stored.tolist() == [0, 0]
    assert not torch.equal(before, m.store.n_stored)


def test_ring_buffer_wraps_and_caps():
    m = _mem(batch=1)
    for i in range(CFG.memory_size * 2 + 3):
        m.commit(torch.randn(1, 4, CFG.d_model) * (i + 1) * 10)
    assert int(m.store.n_stored) <= CFG.memory_size
    assert int(m.store.ptr) < CFG.memory_size


def test_write_gate_rejects_a_repeat_and_accepts_novelty():
    cfg = replace(CFG, write_gate=True, write_threshold=0.3)
    m = HippoMemory(cfg); m.reset(1)
    h = torch.randn(1, 4, cfg.d_model)
    m.commit(h)
    n_after_first = int(m.store.n_stored)
    m.commit(h.clone())                                  # identical -> novelty ~0
    assert int(m.store.n_stored) == n_after_first, "a duplicate segment was stored"
    m.commit(torch.randn(1, 4, cfg.d_model) * 5)         # different -> should store
    assert int(m.store.n_stored) == n_after_first + 1


def test_read_cannot_see_the_current_segments_write():
    """Writes commit at segment boundaries, so a read is only ever over history."""
    m = _mem(batch=1)
    h = torch.randn(1, 6, CFG.d_model)
    m.read_heads["1"].alpha.data.fill_(1.0)
    before = m.read(1, h)
    m.commit(h)                                          # commit AFTER the read
    after = m.read(1, h)
    assert torch.equal(before, h), "read saw a non-empty store before any commit"
    assert not torch.equal(after, h), "the committed segment is not visible next read"


# ------------------------------------------------------------ behaviour ----
def test_zero_init_alpha_is_an_exact_noop():
    """The HC-off baseline must be an EXACT control, not an approximate one."""
    m = _mem()
    m.commit(torch.randn(2, 5, CFG.d_model))
    h = torch.randn(2, 7, CFG.d_model)
    assert torch.equal(m.read(1, h), h)


def test_shapes_and_batch_independence():
    """Example 0's output must not depend on example 1's store."""
    cfg = replace(CFG, readout="graded")
    m = HippoMemory(cfg); m.reset(2)
    m.read_heads["1"].alpha.data.fill_(1.0)
    seg = torch.randn(2, 5, cfg.d_model)
    m.commit(seg)
    h = torch.randn(2, 4, cfg.d_model)
    out_pair = m.read(1, h)
    assert out_pair.shape == h.shape

    solo = HippoMemory(cfg); solo.reset(1)
    solo.load_state_dict(m.state_dict())
    solo.read_heads["1"].alpha.data.fill_(1.0)
    solo.commit(seg[:1])
    out_solo = solo.read(1, h[:1])
    torch.testing.assert_close(out_pair[:1], out_solo, rtol=1e-5, atol=1e-6)


def test_determinism():
    m = _mem()
    m.commit(torch.randn(2, 5, CFG.d_model))
    m.read_heads["1"].alpha.data.fill_(0.5)
    h = torch.randn(2, 4, CFG.d_model)
    assert torch.equal(m.read(1, h), m.read(1, h))


# ------------------------------------------------------------ gradients ----
def test_gradient_reaches_read_params_but_not_the_store():
    m = _mem()
    # Several DISTINCT segments: a store holding one slot is degenerate, since
    # softmax over a single valid entry is exactly 1.0 whatever the score and
    # its derivative is exactly zero. See the test below.
    for i in range(4):
        m.commit(torch.randn(2, 5, CFG.d_model) * (i + 3))
    m.read_heads["1"].alpha.data.fill_(0.5)
    out = m.read(1, torch.randn(2, 4, CFG.d_model, requires_grad=True))
    out.sum().backward()

    head = m.read_heads["1"]
    for name, p in [("dg.proj", head.dg.proj.weight), ("out", head.out.weight),
                    ("gate", head.gate.weight), ("alpha", head.alpha)]:
        assert p.grad is not None and torch.any(p.grad != 0), f"no gradient at {name}"

    assert not m.store.keys.requires_grad
    assert not m.store.values.requires_grad


def test_key_layer_keys_on_the_early_state_and_values_on_the_write_state():
    """key_layer splits a slot across depths: key from the captured early state,
    value from the write block, with value_shift applied to the value only."""
    cfg = replace(PER_TOKEN, key_layer=1, value_shift=0, center_keys=False)
    m = HippoMemory(cfg); m.reset(1)
    early = torch.randn(1, 5, cfg.d_model)
    late = torch.randn(1, 5, cfg.d_model)
    m.pending_key = early
    m.commit(late)
    torch.testing.assert_close(m.store.keys[0, :5], early[0])
    torch.testing.assert_close(m.store.values[0, :5], late[0])
    assert m.pending_key is None, "a captured key must not leak into the next segment"


def test_identity_query_init_survives_full_model_construction():
    """TinyGPT re-initialises every Linear it owns, memory included, so an
    identity query projection must be protected or it silently becomes noise."""
    from mech.model import GPTConfig, TinyGPT
    cfg = replace(PER_TOKEN, query_init="identity", key_layer=1)
    model = TinyGPT(GPTConfig(vocab_size=20, d_model=cfg.d_model, n_layers=4,
                              n_heads=1, seg_len=8), HippoMemory(cfg))
    w = model.memory.read_heads["1"].dg.proj.weight
    assert torch.equal(w, torch.eye(cfg.d_model))


def test_store_save_load_round_trips_exactly_and_refuses_other_models(tmp_path):
    m = HippoMemory(PER_TOKEN); m.reset(2)
    m.commit(torch.randn(2, 6, PER_TOKEN.d_model))
    m.store.save(tmp_path / "s.pt", fingerprint="model-a")

    fresh = HippoMemory(PER_TOKEN); fresh.reset(2)
    fresh.store.load(tmp_path / "s.pt", fingerprint="model-a")
    for f in fresh.store.STATE:
        assert torch.equal(getattr(fresh.store, f), getattr(m.store, f)), f

    h = torch.randn(2, 3, PER_TOKEN.d_model)
    for mem in (m, fresh):
        mem.read_heads["1"].alpha.data.fill_(1.0)
    fresh.load_state_dict(m.state_dict())
    assert torch.equal(m.read(1, h), fresh.read(1, h))

    with pytest.raises(ValueError):
        fresh.store.load(tmp_path / "s.pt", fingerprint="model-b")


def test_position_free_keys_are_token_embeddings_and_match_across_positions():
    """key_pos_free: stored keys are the token embedding alone, so the same token
    at a different position still matches exactly."""
    from mech.model import GPTConfig, TinyGPT
    cfg = replace(PER_TOKEN, key_layer=0, read_layers=(0,), write_layer=1,
                  value_shift=1, query_init="identity", key_pos_free=True,
                  center_keys=False, dg_bypass=True)
    model = TinyGPT(GPTConfig(vocab_size=20, d_model=cfg.d_model, n_layers=2,
                              n_heads=1, seg_len=8), HippoMemory(cfg))
    model.eval()
    model.memory.reset(1)
    seg = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    _, h_write = model(seg)
    model.memory.commit(h_write)
    torch.testing.assert_close(model.memory.store.keys[0, :7], model.tok_emb(seg)[0, :7])

    head = model.memory.read_heads["0"]
    head.stash = {}
    query = torch.tensor([[11, 11, 11, 11, 11, 11, 11, 5]])   # token 5 at position 7, not 2
    model(query)
    assert int(head.stash["w"][0, -1].argmax()) == 2


def test_frozen_identity_query_stays_identity_through_a_training_step():
    from mech.model import GPTConfig, TinyGPT
    cfg = replace(PER_TOKEN, key_layer=0, read_layers=(0,), write_layer=1, value_shift=1,
                  query_init="identity", freeze_query=True, key_pos_free=True)
    model = TinyGPT(GPTConfig(vocab_size=20, d_model=cfg.d_model, n_layers=2,
                              n_heads=1, seg_len=8), HippoMemory(cfg))
    model.memory.read_heads["0"].alpha.data.fill_(1.0)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.memory.reset(1)
    _, hw = model(torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]]))
    model.memory.commit(hw)
    logits, _ = model(torch.tensor([[11, 5, 12, 6, 13, 7, 14, 8]]))
    logits.sum().backward()
    opt.step()
    assert torch.equal(model.memory.read_heads["0"].dg.proj.weight, torch.eye(cfg.d_model))


def test_value_skip_adds_the_earlier_block_to_the_written_value():
    from mech.model import GPTConfig, TinyGPT
    base = replace(PER_TOKEN, key_layer=0, read_layers=(0,), write_layer=2, value_shift=1,
                   query_init="identity", key_pos_free=True, center_keys=False)
    torch.manual_seed(0)
    plain = TinyGPT(GPTConfig(vocab_size=20, d_model=base.d_model, n_layers=3,
                              n_heads=1, seg_len=8), HippoMemory(base))
    skip = TinyGPT(GPTConfig(vocab_size=20, d_model=base.d_model, n_layers=3,
                             n_heads=1, seg_len=8), HippoMemory(replace(base, value_skip_layer=0)))
    skip.load_state_dict(plain.state_dict())
    plain.eval(); skip.eval()
    seg = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        x0 = plain.blocks[0](plain.tok_emb(seg) + plain.pos_emb(torch.arange(8))[None])
        for m in (plain, skip):
            m.memory.reset(1)
            _, hw = m(seg)
            m.memory.commit(hw)
    torch.testing.assert_close(skip.memory.store.values[0, :7],
                               plain.memory.store.values[0, :7] + x0[0, 1:])


def test_normalised_value_skip_balances_the_two_halves():
    """value_skip_norm: value = mean|skip| * (unit(last) + unit(skip)), so neither
    half can dominate by norm."""
    import torch.nn.functional as F
    from mech.model import GPTConfig, TinyGPT
    base = replace(PER_TOKEN, key_layer=0, read_layers=(0,), write_layer=2, value_shift=1,
                   query_init="identity", key_pos_free=True, center_keys=False,
                   value_skip_layer=0)
    torch.manual_seed(0)
    plain = TinyGPT(GPTConfig(vocab_size=20, d_model=base.d_model, n_layers=3,
                              n_heads=1, seg_len=8), HippoMemory(base))
    normed = TinyGPT(GPTConfig(vocab_size=20, d_model=base.d_model, n_layers=3,
                               n_heads=1, seg_len=8), HippoMemory(replace(base, value_skip_norm=True)))
    normed.load_state_dict(plain.state_dict())
    plain.eval(); normed.eval()
    seg = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    with torch.no_grad():
        x = plain.tok_emb(seg) + plain.pos_emb(torch.arange(8))[None]
        skip = plain.blocks[0](x)
        last = plain.blocks[2](plain.blocks[1](skip))
        expect = skip.norm(dim=-1, keepdim=True).mean() * (
            F.normalize(last, dim=-1) + F.normalize(skip, dim=-1))
        normed.memory.reset(1)
        _, hw = normed(seg)
        normed.memory.commit(hw)
    torch.testing.assert_close(normed.memory.store.values[0, :7], expect[0, 1:],
                               rtol=1e-4, atol=1e-5)


def test_key_layer_without_a_captured_state_fails_loudly():
    m = HippoMemory(replace(CFG, key_layer=1)); m.reset(1)
    with pytest.raises(RuntimeError):
        m.commit(torch.randn(1, 5, CFG.d_model))


def test_read_grad_floor_keeps_the_exact_noop_forward():
    """The floor is zero in value, so alpha=0 must still be an exact no-op."""
    m = HippoMemory(replace(CFG, read_grad_floor=0.5)); m.reset(2)
    m.commit(torch.randn(2, 5, CFG.d_model))
    h = torch.randn(2, 7, CFG.d_model)
    assert torch.equal(m.read(1, h), h)


def test_read_grad_floor_reaches_the_query_projection_while_the_gate_is_shut():
    """The gate trap: at alpha=0 the query projection gets NO gradient, so it
    cannot learn to match while memory is unhelpful. The floor must fix exactly
    that and nothing else."""
    def dg_grad(floor):
        torch.manual_seed(0)
        m = HippoMemory(replace(CFG, read_grad_floor=floor)); m.reset(2)
        for i in range(4):
            m.commit(torch.randn(2, 5, CFG.d_model) * (i + 3))
        out = m.read(1, torch.randn(2, 4, CFG.d_model))
        (out * torch.randn_like(out)).sum().backward()
        return m.read_heads["1"].dg.proj.weight.grad

    shut = dg_grad(0.0)
    assert shut is None or torch.all(shut == 0), "alpha=0 already leaks gradient"
    floored = dg_grad(0.5)
    assert floored is not None and torch.any(floored != 0)


def test_write_projections_are_frozen_by_construction():
    """Only detached copies enter the store, so these can never learn.

    Frozen on purpose (WriteHead docstring); learning them needs an auxiliary
    write objective. This test exists so that changing the policy is a
    deliberate act rather than an accident.
    """
    m = _mem()
    assert not m.write_head.to_key.weight.requires_grad
    assert not m.write_head.to_value.weight.requires_grad
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert not any("write_head" in n for n in trainable)


def test_gate_magnitudes_reports_every_read_block():
    cfg = replace(CFG, read_layers=(1, 3))
    m = HippoMemory(cfg); m.reset(1)
    m.read_heads["3"].alpha.data.fill_(0.25)
    g = m.gate_magnitudes()
    assert set(g) == {1, 3}
    assert g[1] == pytest.approx(0.0) and g[3] == pytest.approx(0.25)


def test_single_slot_store_has_no_gradient_by_construction():
    """A one-entry store cannot teach the query projection anything.

    softmax over a single valid slot is exactly 1.0 regardless of the score, so
    its derivative is exactly zero — the same degeneracy that made CA3's
    softmax `confidence` unusable as a gate in v0.1. Recorded as a property, so
    that a future zero-gradient report is diagnosed in seconds rather than
    mistaken for a bug in the layer.
    """
    m = _mem()
    m.commit(torch.randn(2, 5, CFG.d_model))
    assert m.store.n_stored.tolist() == [1, 1]
    head = m.read_heads["1"]; head.alpha.data.fill_(0.5)
    head(torch.randn(2, 4, CFG.d_model), m.store).sum().backward()
    assert torch.all(head.dg.proj.weight.grad == 0)


def test_per_token_writes_fill_one_slot_per_position():
    """Exact content needs per-item slots, not one pooled summary per segment."""
    m = HippoMemory(PER_TOKEN); m.reset(2)
    m.commit(torch.randn(2, 5, PER_TOKEN.d_model))
    # 5 tokens - value_shift(1) = 4 pairs: key from t, value from t+1
    assert m.store.n_stored.tolist() == [4, 4], "expected one slot per (t, t+1) pair"
    m.commit(torch.randn(2, 5, PER_TOKEN.d_model))
    assert m.store.n_stored.tolist() == [8, 8]


def test_value_is_shifted_one_position_from_its_key():
    """Without the shift a matched key returns its OWN content, never what
    follows it, so 'find X then copy the next token' is unrepresentable. The
    prototype writes (dg_codes[j], raw_embeds[j+1]) for exactly this reason."""
    cfg = replace(PER_TOKEN, write_identity=True, value_shift=1)
    m = HippoMemory(cfg); m.reset(1)
    h = torch.randn(1, 6, cfg.d_model)
    m.commit(h)
    torch.testing.assert_close(m.store.keys[0, :5], h[0, :5])
    torch.testing.assert_close(m.store.values[0, :5], h[0, 1:6])


def test_per_token_store_carries_exact_content():
    """The pooled variant cannot do this: distinct tokens must stay separable."""
    cfg = replace(PER_TOKEN, readout="ste", beta=32.0)
    m = HippoMemory(cfg); m.reset(1)
    h = torch.randn(1, 7, cfg.d_model) * 3
    m.commit(h)
    keys = m.store.keys[:, :6]
    sims = torch.nn.functional.normalize(keys, dim=-1) @ \
        torch.nn.functional.normalize(keys, dim=-1).transpose(1, 2)
    off = sims[0] - torch.eye(6)
    assert float(off.abs().max()) < 0.99, "stored token slots collapsed together"
