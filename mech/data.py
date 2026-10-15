"""
data.py — character-level tiny-shakespeare, served as SEGMENTED examples.

Segmentation is the point, not a convenience. The claim under test is a store
that persists across sequence boundaries, and within a single context window
attention already handles everything. So an example is `n_seg` consecutive
segments of `seg_len` tokens: the model sees one segment at a time, the store
carries across them, and anything older than the current segment is reachable
only through memory.
"""

from __future__ import annotations

from pathlib import Path

import torch


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def load(path: str | Path, split: float = 0.9, extra_chars: str = ""):
    """`extra_chars` keeps one vocabulary across tasks, so an LM checkpoint and
    a recall checkpoint stay comparable and interchangeable."""
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer(text + extra_chars)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(ids) * split)
    return tok, ids[:n], ids[n:]


def sample(ids: torch.Tensor, batch: int, n_seg: int, seg_len: int,
           device, generator=None):
    """-> x [B, n_seg, seg_len], y [B, n_seg, seg_len] (next-token targets)."""
    span = n_seg * seg_len + 1
    hi = len(ids) - span
    start = torch.randint(0, hi, (batch,), generator=generator)
    chunk = torch.stack([ids[s:s + span] for s in start])          # [B, span]
    x = chunk[:, :-1].view(batch, n_seg, seg_len)
    y = chunk[:, 1:].view(batch, n_seg, seg_len)
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------
# PG-19: books, kept apart
# --------------------------------------------------------------------------
# A flat concatenation would let one example straddle two books, and then the
# store is asked to carry context across a boundary where there is no context
# to carry — a silent source of "memory does not help". Books stay separate
# tensors and an example is always drawn from inside one of them, so the book
# boundary is structural rather than a sampling accident.


class BookCorpus:
    """A list of per-book id tensors, sampled proportional to length.

    Length-proportional is the point: PG-19 books differ by ~50x in length, so
    picking a book uniformly would oversample the short ones by that factor.
    Sampling by token makes an epoch mean "the corpus", not "the book list".
    """

    def __init__(self, books: list[torch.Tensor]):
        self.books = [b for b in books if len(b) > 0]
        if not self.books:
            raise ValueError("no non-empty books")

    def __len__(self) -> int:
        return len(self.books)

    @property
    def n_tokens(self) -> int:
        return sum(len(b) for b in self.books)

    def flat(self) -> torch.Tensor:
        """All books concatenated, for uses where book boundaries do not matter.

        They matter for the LM task, where the store has to bridge one segment
        to the next and a straddle would ask it to bridge nothing. They do not
        matter for the recall tasks, where prose is only distractor filler drawn
        independently per segment — so those get a flat stream and keep the
        simple `ids[start:start + seg_len]` indexing every task already uses.
        """
        return torch.cat(self.books)

    def sample(self, batch: int, n_seg: int, seg_len: int, device,
               generator=None):
        """-> x, y [B, n_seg, seg_len]. Every example lies inside ONE book."""
        span = n_seg * seg_len + 1
        ok = [b for b in self.books if len(b) >= span]
        if not ok:
            raise ValueError(
                f"no book is long enough for {span} tokens "
                f"(longest is {max(len(b) for b in self.books)})")
        w = torch.tensor([float(len(b)) for b in ok], dtype=torch.double)
        pick = torch.multinomial(w / w.sum(), batch, replacement=True,
                                 generator=generator)
        rows = []
        for p in pick.tolist():
            b = ok[p]
            s = int(torch.randint(0, len(b) - span + 1, (1,),
                                  generator=generator))
            rows.append(b[s:s + span])
        chunk = torch.stack(rows)                                  # [B, span]
        x = chunk[:, :-1].view(batch, n_seg, seg_len)
        y = chunk[:, 1:].view(batch, n_seg, seg_len)
        return x.to(device), y.to(device)


def load_books(root: str | Path, train_split: str = "test",
               val_split: str = "validation", extra_chars: str = "",
               max_books: int | None = None):
    """PG-19 from a directory of `<root>/<split>/*.txt`.

    The split names are PG-19's own, and the defaults are deliberate: the real
    train split is 28,602 books / ~11 GB, far past what this model or this GPU
    budget needs, so the 100-book *test* split is used as the training corpus
    and the 50-book *validation* split as held-out. No book appears in both.
    Naming the directories after PG-19's splits rather than after their use
    here keeps that substitution visible instead of buried.

    The vocabulary is built from the training books only; characters unique to
    the held-out books are dropped by the tokenizer, and the fraction dropped
    is returned so it can be reported rather than assumed negligible.
    """
    root = Path(root)

    def _read(split):
        files = sorted(p for p in (root / split).glob("*.txt"))
        if max_books is not None:
            files = files[:max_books]
        if not files:
            raise FileNotFoundError(f"no .txt books under {root / split}")
        return [p.read_text(encoding="utf-8", errors="replace") for p in files]

    train_txt, val_txt = _read(train_split), _read(val_split)
    tok = CharTokenizer("".join(train_txt) + extra_chars)

    def _enc(texts):
        out, kept, seen = [], 0, 0
        for t in texts:
            ids = tok.encode(t)
            kept += len(ids); seen += len(t)
            out.append(torch.tensor(ids, dtype=torch.long))
        return BookCorpus(out), (1.0 - kept / seen if seen else 0.0)

    train, _ = _enc(train_txt)
    val, val_dropped = _enc(val_txt)
    return tok, train, val, val_dropped
