"""
Tests for episode-aware eviction (r31).

The user's rule from the first planning conversation: "eviction based only
replacing stale incomplete stuff, but not complete episode which is stale". The
mechanical form: slots are owned by episodes, and under pressure the store
evicts free slots first, then INCOMPLETE episodes oldest-first, and only touches
a COMPLETE episode when nothing else is left.

That distinction is the same one sleep uses (r26/r27): a complete episode has a
replay target and can be consolidated, an open one cannot.

    uv run --group sleep pytest tests_mech/ -q
"""

from dataclasses import replace

import torch

from mech.config import MemConfig
from mech.hippo_layer import EpisodicStore

CFG = MemConfig(d_model=8, d_mem=8, memory_size=12, read_layers=(1,),
                write_layer=3, write_per_token=True, episode_evict=True,
                max_episodes=8)


def _store(cfg=CFG, batch=2):
    s = EpisodicStore(cfg)
    s.reset(batch)
    return s


def _write(store, n, tag):
    k = torch.full((store.keys.shape[0], n, store.cfg.d_mem), float(tag))
    store.write_many(k, k.clone())


def test_slots_record_their_episode():
    s = _store()
    s.begin_episode(complete=True)
    _write(s, 4, 1)
    s.end_episode()
    assert int((s.ep_id[0] == 0).sum()) == 4
    assert int((s.ep_id[0] == -1).sum()) == 8      # the rest still free


def test_free_slots_are_used_before_anything_is_evicted():
    s = _store()
    s.begin_episode(); _write(s, 4, 1); s.end_episode()
    s.begin_episode(); _write(s, 4, 2); s.end_episode()
    # both episodes intact, 4 slots still free
    assert int((s.ep_id[0] == 0).sum()) == 4
    assert int((s.ep_id[0] == 1).sum()) == 4
    assert int((s.ep_id[0] == -1).sum()) == 4


def test_a_complete_episode_survives_an_overflow_that_evicts_an_incomplete_one():
    s = _store()
    s.begin_episode(complete=True); _write(s, 6, 1); s.end_episode()   # ep 0: complete
    s.begin_episode();              _write(s, 6, 2); s.end_episode()   # ep 1: incomplete
    assert int((s.ep_id[0] == -1).sum()) == 0                          # store full
    s.begin_episode(); _write(s, 6, 3); s.end_episode()                # ep 2 needs room
    # ep 1 (incomplete) is gone; ep 0 (complete) is untouched
    assert int((s.ep_id[0] == 0).sum()) == 6
    assert int((s.ep_id[0] == 1).sum()) == 0
    assert int((s.ep_id[0] == 2).sum()) == 6


def test_the_ring_buffer_would_have_evicted_the_complete_one():
    """Same sequence without the policy: the oldest slots go, facts included."""
    s = _store(replace(CFG, episode_evict=False))
    s.begin_episode(complete=True); _write(s, 6, 1); s.end_episode()
    s.begin_episode();              _write(s, 6, 2); s.end_episode()
    s.begin_episode();              _write(s, 6, 3); s.end_episode()
    # ring order overwrote episode 0's slots, which held the complete episode
    assert int((s.ep_id[0] == 0).sum()) == 0


def test_complete_episodes_are_evicted_only_when_nothing_else_remains():
    s = _store()
    s.begin_episode(complete=True); _write(s, 6, 1); s.end_episode()
    s.begin_episode(complete=True); _write(s, 6, 2); s.end_episode()
    s.begin_episode(complete=True); _write(s, 6, 3); s.end_episode()
    # everything is complete, so the oldest complete episode yields
    assert int((s.ep_id[0] == 0).sum()) == 0
    assert int((s.ep_id[0] == 2).sum()) == 6


def test_default_is_the_old_ring_behaviour():
    s = _store(replace(CFG, episode_evict=False))
    s.begin_episode(); _write(s, 4, 1); s.end_episode()
    assert int(s.ptr[0]) == 4


def test_validity_follows_ownership_not_a_prefix():
    """With holes in the store, `idx < n_stored` marks the wrong slots.

    The ring buffer fills in order so a prefix test works. Episode eviction
    writes wherever the policy frees room, so validity has to be ownership.
    """
    s = _store()
    s.begin_episode(complete=True); _write(s, 6, 1); s.end_episode()
    s.begin_episode();              _write(s, 6, 2); s.end_episode()
    s.begin_episode();              _write(s, 6, 3); s.end_episode()   # evicts ep 1
    m = s.valid_mask()
    assert bool((m[0] == (s.ep_id[0] >= 0)).all())
    assert int(m[0].sum()) == 12                    # ep 0 and ep 2, ep 1 replaced


def test_ring_mode_keeps_the_prefix_mask():
    s = _store(replace(CFG, episode_evict=False))
    _write(s, 5, 1)
    assert int(s.valid_mask()[0].sum()) == 5


def test_the_slot_diagnostic_is_skipped_when_its_assumptions_break():
    """It maps slot p to position p, which selection, eviction and a small store
    all violate — and indexing past the store crashed r31's M=64 runs."""
    import types
    from mech.train import diag_applies

    def fake(cfg):
        return types.SimpleNamespace(memory=types.SimpleNamespace(cfg=cfg))

    base = replace(CFG, episode_evict=False, memory_size=256)
    assert diag_applies(fake(base), 128)                       # plain ring, big store
    assert not diag_applies(fake(replace(base, episode_evict=True)), 128)
    assert not diag_applies(fake(replace(base, write_select="surprisal")), 128)
    assert not diag_applies(fake(replace(base, memory_size=64)), 128)
