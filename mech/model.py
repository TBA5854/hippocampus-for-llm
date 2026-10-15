"""
model.py — a small causal LM with the episodic store wired into the stack.

Per block:      x -> gated-HC read -> attn -> MLP

The read sits BEFORE the block's attention so recalled content participates in
attention's own mixing rather than being appended after it. The write is taken
from a later block, because that representation is the most complete thing to
store; see mech/hippo_layer.py for why read and write do not share a space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mech.hippo_layer import HippoMemory


@dataclass
class GPTConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    seg_len: int = 128
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(D, dim=2)
        shape = (B, S, self.n_heads, D // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig, memory: HippoMemory | None = None):
        super().__init__()
        self.cfg = cfg
        self.memory = memory
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seg_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight            # tied
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            if getattr(m.weight, "_keep_init", False):
                return          # a deliberate init (e.g. identity query projection)
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        """idx [B, S] -> logits [B, S, V], write-layer hidden state [B, S, D]."""
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device)
        tok = self.tok_emb(idx)
        x = tok + self.pos_emb(pos)[None]
        if self.memory is not None:
            self.memory.cur_tok = idx                 # r64: tokens for the pool/index
            self.memory.pool_cache = None             # rebuild the pool each forward;
                                                      # reuse is ACROSS LAYERS, not segments
            self.memory.hop_q = None                  # r68: hop chain restarts each forward
            # r75: token embedding for the pointer decode. Use object.__setattr__ so
            # nn.Module does NOT register this shared Parameter (registering it would
            # duplicate tok_emb.weight into the state_dict as memory.tok_emb_ref and
            # break warm-starts from such a checkpoint).
            object.__setattr__(self.memory, "tok_emb_ref", self.tok_emb.weight)
        h_write = None
        wlayers = self.memory.cfg.write_layers if self.memory is not None else None
        per_layer = self.memory is not None and getattr(self.memory, "per_layer", False)
        shared_pl = self.memory is not None and getattr(self.memory, "shared_pl", False)
        wcapture = []                                     # per-layer states (r56)
        pos_free = self.memory is not None and self.memory.cfg.key_pos_free
        keytok = tok
        if pos_free and getattr(self.memory.cfg, "fine_key", False):
            # r89: position-free BIGRAM key = emb(token_t) + alpha*emb(token_{t-1}).
            # The query has the same (context, key) adjacency, so it reproduces the
            # composite exactly; facts sharing a key token are separated by context.
            prev = torch.zeros_like(tok)
            prev[:, 1:] = tok[:, :-1]
            keytok = tok + self.memory.cfg.fine_key_alpha * prev
        for i, blk in enumerate(self.blocks):
            if self.memory is not None:
                q = keytok if (pos_free and i == 0) else None
                if i == self.memory.cfg.key_layer:
                    # Captured BEFORE the read, so a key is exactly the state a
                    # query at this depth is computed from, not one that already
                    # carries recalled content.
                    self.memory.pending_key = q if q is not None else x
                x = self.memory.read(i, x, q=q)           # read before attention
            x = blk(x)
            if self.memory is not None and i == self.memory.cfg.value_skip_layer:
                skip = x
            if self.memory is not None and i == self.memory.cfg.write_layer:
                h_write = x
            if (per_layer or shared_pl) and i in self.memory._layers:
                self.memory.pending_values[i] = x.detach()  # this layer's own value;
                # DETACHED like every other write path (train loop detaches h_write).
                # Without this the store holds live-graph tensors and a later segment's
                # read links back through every earlier segment's write -> the graph is
                # retained across all segments and OOMs (r104). Stored content never
                # receives gradient by design, so detaching is correct, not a loss.
            elif wlayers is not None and i in wlayers:
                wcapture.append(x)
        if wlayers is not None and wcapture:
            # Write at every listed layer: the value is the mean of those depths,
            # keeping the slot count identical to a single-layer write.
            h_write = torch.stack(wcapture, 0).mean(0)
        if shared_pl and h_write is None:
            # r103: shared-per-layer has no single write layer. commit() ignores this
            # value (it writes each layer's own pending_values into the shared store),
            # but the train loop only calls commit when h_write is not None.
            h_write = x
        if self.memory is not None and self.memory.cfg.value_skip_layer is not None \
                and h_write is not None:
            if self.memory.cfg.value_skip_norm:
                scale = skip.norm(dim=-1, keepdim=True).mean()
                h_write = scale * (F.normalize(h_write, dim=-1) + F.normalize(skip, dim=-1))
            else:
                h_write = h_write + skip
        logits = self.head(self.ln_f(x))
        # Write-rate gating (r30) ranks positions by how surprising the token a
        # slot would RETURN is: slot i pairs key i with the token at i+shift, so
        # a slot is worth spending when that token could not have been predicted.
        # Stashed rather than returned, so every existing caller keeps working.
        self.write_score = None
        mcfg = getattr(self.memory, "cfg", None) if self.memory is not None else None
        if mcfg is not None and getattr(mcfg, "write_select", "all") == "surprisal":
            sh = max(int(mcfg.value_shift), 1)
            if idx.shape[1] > sh:
                tgt = idx[:, sh:]                                   # [B, S-sh]
                lp = F.log_softmax(logits[:, :idx.shape[1] - sh].float(), dim=-1)
                self.write_score = -lp.gather(2, tgt[..., None]).squeeze(2).detach()
        return logits, h_write

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def fingerprint(self) -> str:
        """Digest of every learned weight. A persisted store is only meaningful to
        the exact weights that wrote it, so this is what a store file is keyed to.
        Buffers are excluded: key_mean drifts at eval time and travels with the
        store instead."""
        import hashlib
        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
        return h.hexdigest()
