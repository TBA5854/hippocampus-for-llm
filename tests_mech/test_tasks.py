"""
Tests for the MQAR task builder, specifically the key-vocabulary size.

`N_SYMBOLS = 32` was a module constant until r22, which means every result up to
r21 was run at 32 distinct keys — and with `--n-facts 32` that is *every* key used
once, so fact count and key vocabulary were the same number. `--n-symbols` splits
them. The default must stay 32 or the whole round history stops being comparable.

    uv run --group sleep pytest tests_mech/ -q
"""

import torch

from mech.tasks import N_SYMBOLS, mqar_batch, mqar_vocab, _sym

BASE = 80                      # TinyStories ASCII vocabulary
IDS = torch.randint(0, BASE, (4096,))


def _batch(n_symbols, n_facts, tile=False):
    g = torch.Generator().manual_seed(0)
    return mqar_batch(IDS, BASE, batch=2, n_seg=3, seg_len=32, n_facts=n_facts,
                      device="cpu", generator=g, tile=tile, n_symbols=n_symbols)


def test_default_is_unchanged():
    assert N_SYMBOLS == 32
    assert mqar_vocab(BASE) == mqar_vocab(BASE, 32) == BASE + 65
    assert _sym(BASE) == _sym(BASE, 32)


def test_vocab_scales_with_symbols():
    # base + N keys + N values + 1 query marker
    assert mqar_vocab(BASE, 128) == BASE + 2 * 128 + 1
    kbase, vbase, qtok = _sym(BASE, 128)
    assert (kbase, vbase, qtok) == (BASE, BASE + 128, BASE + 256)


def test_symbols_stay_inside_their_ranges():
    n = 128
    x, y, mask = _batch(n, n_facts=64)
    vocab = mqar_vocab(BASE, n)
    assert int(x.max()) < vocab and int(y.max()) < vocab

    kbase, vbase, qtok = _sym(BASE, n)
    facts = x[:, 0]                                   # segment 0 holds the pairs
    syms = facts[facts >= kbase]
    assert int(syms.min()) >= kbase and int(syms.max()) < qtok

    # the answer is one token, and it is a value symbol
    answers = y[mask]
    assert answers.numel() == x.shape[0]
    assert bool(((answers >= vbase) & (answers < qtok)).all())


def test_more_symbols_allow_more_distinct_keys():
    # 64 facts need at least 64 distinct keys: impossible at the 32-symbol default.
    # seg_len must hold 2 tokens per pair, or segment 0 truncates the list.
    g = torch.Generator().manual_seed(0)
    x, _, _ = mqar_batch(IDS, BASE, batch=2, n_seg=3, seg_len=160, n_facts=64,
                         device="cpu", generator=g, tile=False, n_symbols=128)
    kbase, _, qtok = _sym(BASE, 128)
    row = x[0, 0]
    keys = row[(row >= kbase) & (row < kbase + 128)]
    assert len(set(keys.tolist())) >= 64
