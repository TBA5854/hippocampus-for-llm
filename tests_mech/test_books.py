"""
Tests for the PG-19 book corpus (r34).

The whole claim under test on PG-19 is that the store carries usable context
ACROSS segment boundaries. That claim is only measurable if the segments of one
example actually belong together — so the property that matters most here is
that an example never straddles two books. A flat concatenation gets this wrong
silently: the loss looks bad and the conclusion "memory does not help" is drawn
from a sampling bug.

    uv run --group sleep pytest tests_mech/ -q
"""

from pathlib import Path

import pytest
import torch

from mech.data import BookCorpus, load_books


def _books(lengths, span_fill=True):
    """Book i is filled entirely with token id i, so a straddle is visible."""
    return [torch.full((n,), i, dtype=torch.long)
            for i, n in enumerate(lengths)]


def test_an_example_never_straddles_two_books():
    c = BookCorpus(_books([200, 200, 200]))
    x, y = c.sample(batch=64, n_seg=4, seg_len=16, device="cpu")
    # every row is drawn from one book, so every row is one constant id
    for row in x.reshape(64, -1):
        assert int(row.min()) == int(row.max())


def test_shapes_and_next_token_alignment():
    ids = torch.arange(500, dtype=torch.long)
    c = BookCorpus([ids])
    x, y = c.sample(batch=2, n_seg=3, seg_len=8, device="cpu")
    assert x.shape == (2, 3, 8) and y.shape == (2, 3, 8)
    # y is x shifted by one across the whole flattened span, segments included
    assert torch.equal(y.reshape(2, -1)[:, :-1], x.reshape(2, -1)[:, 1:])


def test_books_are_sampled_in_proportion_to_length():
    """Uniform-per-book would oversample the short ones ~50x on real PG-19."""
    c = BookCorpus(_books([100, 100, 10000]))
    x, _ = c.sample(batch=400, n_seg=1, seg_len=8, device="cpu",
                    generator=torch.Generator().manual_seed(0))
    share = (x.reshape(400, -1)[:, 0] == 2).double().mean()
    assert share > 0.9          # book 2 is ~98% of the tokens


def test_a_span_longer_than_every_book_is_an_error_not_a_silent_truncation():
    c = BookCorpus(_books([64, 64]))
    with pytest.raises(ValueError, match="long enough"):
        c.sample(batch=2, n_seg=4, seg_len=128, device="cpu")


def test_short_books_are_skipped_rather_than_straddled():
    """One book is too short for the span; it must simply not be drawn."""
    c = BookCorpus(_books([32, 5000]))
    x, _ = c.sample(batch=32, n_seg=2, seg_len=64, device="cpu")
    assert int(x.min()) == 1 and int(x.max()) == 1


def test_load_books_builds_the_vocab_from_training_books_only(tmp_path):
    """A character that appears only in held-out text must not enter the
    vocabulary — and the fraction of held-out text it costs is reported, so
    "negligible" is a measurement rather than an assumption."""
    (tmp_path / "test").mkdir()
    (tmp_path / "validation").mkdir()
    (tmp_path / "test" / "a.txt").write_text("abcabcabc")
    (tmp_path / "validation" / "b.txt").write_text("abcZ")

    tok, train, val, dropped = load_books(tmp_path)
    assert "Z" not in tok.stoi
    assert len(train) == 1 and len(val) == 1
    assert dropped == pytest.approx(0.25)       # 1 of 4 held-out characters
    assert val.n_tokens == 3


def test_max_books_limits_both_splits(tmp_path):
    for split in ("test", "validation"):
        (tmp_path / split).mkdir()
        for i in range(3):
            (tmp_path / split / f"{i}.txt").write_text("hello world " * 10)
    _, train, val, _ = load_books(tmp_path, max_books=2)
    assert len(train) == 2 and len(val) == 2
