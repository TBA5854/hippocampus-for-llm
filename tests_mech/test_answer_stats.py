"""
Tests for the character-level answer metrics (r35).

Why these exist: the note that created MQAR records that exact match at
character level is too blunt an instrument -- the answer is two characters, so a
real retrieval edge (CE 3.148 against a 3.256 scrambled control, chance
log(26) = 3.258) "almost never produced a correct pair". Exact match read ~0
while the model genuinely knew something, and the whole project moved to
single-token symbols because of it.

So a character-level language round must report per-character accuracy and
answer-position CE as well, or a negative result cannot distinguish "the store
does not carry this" from "the ruler is too coarse".

    uv run --group sleep pytest tests_mech/ -q
"""

import math

import torch

from mech.tasks import answer_stats, exact_match


def _logits(vocab, target_rows, confidence=10.0):
    """One-hot-ish logits putting `confidence` on each given target id."""
    B, S = len(target_rows), len(target_rows[0])
    lg = torch.zeros(B, S, vocab)
    for b, row in enumerate(target_rows):
        for t, v in enumerate(row):
            if v >= 0:
                lg[b, t, v] = confidence
    return lg


def test_char_accuracy_sees_a_half_right_answer_that_exact_match_calls_zero():
    """The exact failure that made this metric necessary."""
    vocab = 30
    y = torch.tensor([[5, 6], [7, 8]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    # every example gets the FIRST character right and the second wrong
    lg = _logits(vocab, [[5, 0], [7, 0]])

    hits, tot = exact_match(lg, y, mask)
    assert (hits, tot) == (0, 2)            # exact match: nothing

    out = {}
    answer_stats(lg, y, mask, out)
    assert out["char_hits"] == 2 and out["char_total"] == 4   # half the chars
    assert out["char_hits"] / out["char_total"] == 0.5


def test_answer_ce_is_measured_only_on_masked_positions():
    vocab = 26
    y = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, False]])
    lg = torch.zeros(1, 3, vocab)           # uniform -> CE = log(26) everywhere
    out = {}
    answer_stats(lg, y, mask, out)
    assert out["ce_n"] == 1                 # one position, not three
    assert out["ce_sum"] / out["ce_n"] == torch.tensor(float(vocab)).log().item()


def test_a_uniform_model_sits_at_the_chance_reference():
    """log(26) = 3.258 is the number the MQAR note quotes as 'knows nothing'."""
    out = {}
    answer_stats(torch.zeros(4, 2, 26), torch.randint(0, 26, (4, 2)),
                 torch.ones(4, 2, dtype=torch.bool), out)
    assert abs(out["ce_sum"] / out["ce_n"] - math.log(26)) < 1e-5


def test_accumulates_across_calls():
    """evaluate() folds many batches into one dict."""
    out = {}
    for _ in range(3):
        answer_stats(torch.zeros(2, 2, 26), torch.zeros(2, 2, dtype=torch.long),
                     torch.ones(2, 2, dtype=torch.bool), out)
    assert out["char_total"] == 12 and out["ce_n"] == 12


def test_an_empty_mask_adds_nothing():
    """Segments without an answer must not dilute the averages."""
    out = {"char_hits": 1, "char_total": 2}
    answer_stats(torch.zeros(2, 2, 26), torch.zeros(2, 2, dtype=torch.long),
                 torch.zeros(2, 2, dtype=torch.bool), out)
    assert out == {"char_hits": 1, "char_total": 2}


def test_per_position_accuracy_separates_the_two_answer_characters():
    """r36. r35 measured 47.1% of characters right but only 10.0% of pairs, so
    the two do not fail independently (0.471^2 = 22% would be). They come from
    DIFFERENT slots -- with value_shift 1 the slot keyed on ':' returns the
    first answer character and the slot keyed on that character returns the
    second -- so which one fails localises the loss."""
    y = torch.tensor([[0, 0, 7, 9], [0, 0, 7, 9]])
    mask = torch.tensor([[False, False, True, True]] * 2)
    # first answer character right in both rows, second wrong in both
    lg = _logits(30, [[0, 0, 7, 1], [0, 0, 7, 1]])
    out = {}
    answer_stats(lg, y, mask, out)
    assert out["pos0_hits"] == 2 and out["pos0_total"] == 2      # 100%
    assert out["pos1_hits"] == 0 and out["pos1_total"] == 2      # 0%
    assert out["char_hits"] == 2 and out["char_total"] == 4      # 50% overall
