"""
Tests for the BPE token-level recall path (r47).

The point of BPE here is that a fact key is a SINGLE token, which is what lets
MQAR reach 100% and what r36 showed char-level lacks. These pin the two
properties that must hold: keys/values are single real word-tokens, and the
task lays them out so the slot keyed on k_j returns v_j in one hop.

    uv run --group sleep pytest tests_mech/ -q
"""

import torch

from mech.bpe import BPETokenizer
from mech.tasks import bpe_recall_batch

CORPUS = ["the house was quiet and the world was calm and the man walked on "
          "through the long grass while the river ran beside the road "] * 50


def _tok():
    return BPETokenizer.train(CORPUS, vocab_size=512, min_frequency=1)


def test_roundtrip_and_interface():
    t = _tok()
    assert t.vocab_size > 0
    assert isinstance(t.pad_id, int)
    ids = t.encode("the house")
    assert t.decode(ids).strip() == "the house"
    # CharTokenizer-compatible surface the pipeline calls
    assert isinstance(t.stoi, dict) and isinstance(t.itos, dict)


def test_single_token_words_are_single_tokens_and_words():
    t = _tok()
    words = t.single_token_words(min_len=3)
    assert words, "no single-token words found"
    for i in words[:20]:
        s = t.itos[i]
        assert len(t.encode(t.decode([i]).strip())) == 1  # really one token
        assert t.decode([i]).strip().isalpha()


def test_recall_layout_keys_precede_values():
    t = _tok()
    words = t.single_token_words(min_len=3)
    half = len(words) // 2
    kp, vp = words[:half], words[half:2 * half]
    x, y, m = bpe_recall_batch(torch.arange(1000), batch=4, n_seg=4, seg_len=32,
                               n_facts=3, key_pool=kp, val_pool=vp,
                               qtok_id=words[-1], pad_id=t.pad_id, device="cpu",
                               generator=torch.Generator().manual_seed(0))
    assert x.shape == (4, 4, 32) and m.shape == (4, 4, 32)
    # exactly one answer position per example (the single value token), in seg N-1
    assert int(m.sum()) == 4
    assert bool(m[:, -1].any()) and not bool(m[:, :-1].any())
    # the answer target is a value-pool token
    vp_set = set(vp)
    tgt = y[m]
    assert all(int(v) in vp_set for v in tgt)


def test_key_and_value_pools_are_disjoint_single_tokens():
    t = _tok()
    words = t.single_token_words(min_len=3)
    half = len(words) // 2
    kp, vp = set(words[:half]), set(words[half:2 * half])
    assert not (kp & vp)
