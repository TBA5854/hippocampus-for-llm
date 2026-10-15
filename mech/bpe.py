"""
bpe.py — a BPE tokenizer with the CharTokenizer interface, for token-level recall.

Why this exists: r36 found the language recall ceiling is an addressing constraint.
A fact key written in char-level text (`ab`) is TWO tokens, and the only token
available at the retrieval point is the shared delimiter `:`, so the read is a
uniform draw over the fact set (pos0 = 1/n_facts). r44 confirmed the residual
gap-to-ceiling grows with fact count and training does not close it — it is the
multi-token-key tax.

A subword (BPE) vocabulary makes a key a SINGLE token, which is exactly the
property that lets MQAR reach 100% (its symbols are single tokens). The open
question BPE answers that MQAR could not: MQAR keys are synthetic ids OUTSIDE the
text vocabulary, so they never compete with real word embeddings. Here keys,
values and distractor prose all live in one real-language vocabulary, so a key
competes with the words the model already predicts — the honest test of whether
single-token addressing survives a shared, correlated embedding space.

The class mirrors CharTokenizer (vocab_size / encode / decode / stoi / itos) so
data.py and tasks.py consume it unchanged, plus `pad_id` and `single_token_words`
which the token-level task needs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

PAD = "<pad>"


class BPETokenizer:
    """Byte-level BPE trained on a corpus, exposing the CharTokenizer interface."""

    def __init__(self, tk: Tokenizer):
        self._tk = tk
        self.stoi = tk.get_vocab()                     # token string -> id
        self.itos = {i: s for s, i in self.stoi.items()}
        self.pad_id = self.stoi[PAD]

    # --- CharTokenizer-compatible surface ---------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def encode(self, s: str) -> list[int]:
        return self._tk.encode(s).ids

    def decode(self, ids) -> str:
        return self._tk.decode([int(i) for i in ids])

    # --- training / persistence -------------------------------------------
    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 8192,
              min_frequency: int = 2) -> "BPETokenizer":
        tk = Tokenizer(BPE(unk_token="<unk>"))
        tk.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tk.decoder = __import__("tokenizers").decoders.ByteLevel()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                             special_tokens=[PAD, "<unk>"])
        tk.train_from_iterator(texts, trainer=trainer)
        return cls(tk)

    def save(self, path: str | Path) -> None:
        self._tk.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # --- the token-level recall task needs a pool of single-token words ----
    def single_token_words(self, min_len: int = 3, max_len: int = 12,
                           limit: int | None = None) -> list[int]:
        """Ids of vocab entries that decode to one clean lowercase word.

        A recall key/value must be ONE token AND a real word. Byte-level BPE
        writes a leading space as 'Ġ', so a word-initial token looks like
        'Ġhouse'; we keep those, strip the marker, and require the rest to be
        alphabetic. Sorted by id (frequent merges first) so a smaller `limit`
        yields commoner words.
        """
        out = []
        for s, i in sorted(self.stoi.items(), key=lambda kv: kv[1]):
            w = s[1:] if s.startswith("Ġ") else None
            if w and w.isalpha() and w.islower() and min_len <= len(w) <= max_len:
                out.append(i)
                if limit and len(out) >= limit:
                    break
        return out


def load_books_bpe(root: str | Path, tokenizer: BPETokenizer,
                   train_split: str = "test", val_split: str = "validation",
                   max_books: int | None = None):
    """PG-19 loaded and BPE-encoded, returning flat id tensors (train, val).

    Mirrors data.load_books but with a pre-trained BPE tokenizer and no vocab
    building (the tokenizer already fixes the vocabulary). Flat streams, since
    the token-level recall task uses prose only as per-segment distractor filler.
    """
    root = Path(root)

    def _read(split):
        files = sorted((root / split).glob("*.txt"))
        if max_books is not None:
            files = files[:max_books]
        if not files:
            raise FileNotFoundError(f"no .txt under {root / split}")
        return [p.read_text(encoding="utf-8", errors="replace") for p in files]

    def _enc(texts):
        ids = []
        for t in texts:
            ids.extend(tokenizer.encode(t))
        return torch.tensor(ids, dtype=torch.long)

    return _enc(_read(train_split)), _enc(_read(val_split))
