"""
hippo_layer.py — the episodic store as a trained layer.

    x -> gated-HC -> attn -> MLP

Ported from the PyTorch prototype (brain/side-quest/homeostasis: dg.py, ca1.py,
ca3.py) with four changes that the prototype's design could not support:

  1. BATCHED. The prototype is [seq, dim] throughout and its transformer asserts
     B == 1. Everything here carries a leading batch dim, and the store is
     per-example.
  2. BUFFER-BACKED. The prototype's MemoryBank is a plain Python object holding
     raw tensors, so .to()/state_dict/compile all break (its BertReasoner has a
     workaround comment about exactly this). Here the store is registered
     buffers.
  3. READ AND WRITE SPLIT ACROSS DEPTHS. Write late, read early — the late
     representation is the most complete thing to store, and an early read
     leaves the rest of the stack able to use what came back. Reading at the
     last block is the hidden-state fusion v0.2 measured at 40%.
     Because layer-1 and layer-7 hidden states are not comparable, the store
     gets its OWN space (d_mem) and both sides project into it. That is what
     attention already does with W_Q/W_K.
  4. WRITES COMMIT AT SEGMENT BOUNDARIES, so a read within a segment only ever
     sees earlier segments. The prototype needs a causal mask over the store
     precisely because it reads and writes inside one pass; this does not.

Writes are detached: no backprop through memory contents (Memorizing
Transformers does the same). Gradient reaches the projections and gates only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mech.config import MemConfig


# ---------------------------------------------------------------- store ----
class EpisodicStore(nn.Module):
    """Fixed-capacity ring buffer of episode summaries, one store per example."""

    def __init__(self, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        # persistent=False: this is transient per-example state, not a learned
        # parameter, so it must not land in state_dict — but it must still
        # follow .to(device), which a plain attribute would not.
        self.register_buffer("keys", torch.zeros(1, cfg.memory_size, cfg.d_mem),
                             persistent=False)
        self.register_buffer("values", torch.zeros(1, cfg.memory_size, cfg.d_mem),
                             persistent=False)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer("n_stored", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        # Running mean of written keys. Persistent: it is calibration, not
        # transient episode state, and a restored model needs it.
        # --- episodes (r31) --------------------------------------------------
        # ep_id[b, slot] = which episode owns the slot, -1 = free. Eviction is
        # by episode, not by age: the user's rule is "replace stale INCOMPLETE
        # episodes, never a complete one", and completeness is what makes an
        # episode consolidatable (r26: a closed episode has a replay target).
        self.register_buffer("ep_id", torch.full((1, cfg.memory_size), -1,
                                                 dtype=torch.long), persistent=False)
        self.register_buffer("ep_complete", torch.zeros(1, cfg.max_episodes,
                                                        dtype=torch.bool), persistent=False)
        self.register_buffer("ep_next", torch.zeros(1, dtype=torch.long), persistent=False)
        # Which SOURCE POSITION each slot came from. r32: a content-based write
        # policy is a function of the weights, so recomputing the selection after
        # sleep picks a different subset and the re-encode collapses (6.3%).
        # Recording it means an episode can be rewritten exactly as it was.
        self.register_buffer("slot_pos", torch.full((1, cfg.memory_size), -1,
                                                    dtype=torch.long), persistent=False)
        # r64: the KEY TOKEN id per slot, for the CSA2 token-index candidate pool
        # (a query looks up slots whose stored token matches its own). -1 = free.
        self.register_buffer("key_tok", torch.full((1, cfg.memory_size), -1,
                                                   dtype=torch.long), persistent=False)
        # r67: GLOBAL write order per slot (monotonic across segments), for the
        # recency read. slot_pos is within-segment and repeats each segment, so it
        # cannot say which of two same-key slots is newer; this can. -1 = free.
        self.register_buffer("slot_seq", torch.full((1, cfg.memory_size), -1,
                                                    dtype=torch.long), persistent=False)
        self.register_buffer("write_count", torch.zeros(1, dtype=torch.long),
                             persistent=False)
        # r76: the VALUE token id per slot, so the gated controller can tell an
        # update (same key, different value) from a skip (same key AND value).
        self.register_buffer("value_tok", torch.full((1, cfg.memory_size), -1,
                                                     dtype=torch.long), persistent=False)
        self.register_buffer("key_mean", torch.zeros(cfg.d_mem))
        self.register_buffer("mean_ready", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def reset(self, batch_size: int, device=None, dtype=None) -> None:
        device = device or self.keys.device
        dtype = dtype or self.keys.dtype
        M, D = self.cfg.memory_size, self.cfg.d_mem
        self.keys = torch.zeros(batch_size, M, D, device=device, dtype=dtype)
        self.values = torch.zeros(batch_size, M, D, device=device, dtype=dtype)
        self.ptr = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.n_stored = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.ep_id = torch.full((batch_size, M), -1, dtype=torch.long, device=device)
        self.ep_complete = torch.zeros(batch_size, self.cfg.max_episodes,
                                       dtype=torch.bool, device=device)
        self.ep_next = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.slot_pos = torch.full((batch_size, M), -1, dtype=torch.long, device=device)
        self.key_tok = torch.full((batch_size, M), -1, dtype=torch.long, device=device)
        self.slot_seq = torch.full((batch_size, M), -1, dtype=torch.long, device=device)
        self.write_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.value_tok = torch.full((batch_size, M), -1, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ episodes
    @torch.no_grad()
    def begin_episode(self, complete: bool = False) -> None:
        """Open an episode. Slots written from now on belong to it."""
        if int(self.ep_next.max()) >= self.cfg.max_episodes:
            # wrap the episode counter; the oldest ids simply stop being referenced
            self.ep_next.zero_()
        if complete:
            self.mark_complete()

    @torch.no_grad()
    def mark_complete(self) -> None:
        """An episode with its response present: consolidatable, so evict last."""
        b = torch.arange(self.ep_complete.shape[0], device=self.ep_complete.device)
        self.ep_complete[b, self.ep_next] = True

    @torch.no_grad()
    def end_episode(self) -> None:
        self.ep_next = (self.ep_next + 1) % self.cfg.max_episodes

    @torch.no_grad()
    def _alloc(self, S: int) -> Tensor:
        """Choose S slots per example. Ring order, or by episode priority.

        Priority (lowest number is taken first):
            free slots            -> -1
            INCOMPLETE episodes   -> their episode id (oldest first)
            COMPLETE episodes     -> id + max_episodes (only once nothing else is left)

        That is the user's rule made mechanical: a complete episode survives an
        overflow that a stale incomplete one does not.
        """
        B, M = self.ep_id.shape
        if not self.cfg.episode_evict:
            if getattr(self.cfg, "store_grow", False):
                self._grow(S)                          # append like a KV cache; no lapping
                M = self.keys.shape[1]                 # capacity now >= ptr + S
            idx = (self.ptr[:, None] + torch.arange(S, device=self.keys.device)[None, :]) % M
            return idx
        ep = self.ep_id
        complete = torch.zeros_like(ep, dtype=torch.bool)
        valid = ep >= 0
        if bool(valid.any()):
            safe = ep.clamp_min(0)
            complete = torch.gather(self.ep_complete, 1, safe) & valid
        pri = torch.where(valid, ep, torch.full_like(ep, -1))
        pri = pri + complete.long() * self.cfg.max_episodes
        return pri.topk(S, dim=1, largest=False).indices

    @torch.no_grad()
    def _grow(self, need: int) -> None:
        """Grow the store like a KV cache: if the next `need` slots would overflow the
        current capacity, double it (amortized O(1)) and copy — the plain write path then
        appends without ever lapping (r100). Grows every per-slot buffer."""
        M = self.keys.shape[1]
        if int(self.ptr.max()) + need <= M:
            return
        B, D, dev = self.keys.shape[0], self.keys.shape[2], self.keys.device
        newM = max(2 * M, int(self.ptr.max()) + need)
        def g3(t):
            n = torch.zeros(B, newM, t.shape[2], dtype=t.dtype, device=dev); n[:, :M] = t; return n
        def g2(t, fill):
            n = torch.full((B, newM), fill, dtype=t.dtype, device=dev); n[:, :M] = t; return n
        self.keys, self.values = g3(self.keys), g3(self.values)
        self.ep_id = g2(self.ep_id, -1)
        self.slot_pos = g2(self.slot_pos, -1)
        self.key_tok = g2(self.key_tok, -1)
        self.slot_seq = g2(self.slot_seq, -1)
        self.value_tok = g2(self.value_tok, -1)

    STATE = ("keys", "values", "ptr", "n_stored", "key_mean", "mean_ready")

    def save(self, path, fingerprint: str) -> None:
        """Persist the store. It holds codes only the model that wrote them can
        read (keys are that model's layer states, values its last block), so the
        file is stamped with the model's fingerprint and refuses any other.
        key_mean travels with it: writes update the centring mean even at eval
        time, so the checkpoint's copy is stale for any store written after it."""
        torch.save({"fingerprint": fingerprint,
                    **{f: getattr(self, f).detach().cpu() for f in self.STATE}}, path)

    def load(self, path, fingerprint: str, device=None) -> None:
        blob = torch.load(path, map_location="cpu")
        if blob["fingerprint"] != fingerprint:
            raise ValueError(
                f"store was written by model {blob['fingerprint'][:12]}, not "
                f"{fingerprint[:12]}; its keys and values are unreadable here")
        device = device or self.keys.device
        for f in self.STATE:
            t = blob[f].to(device)
            if f in ("key_mean", "mean_ready"):      # registered persistent buffers
                getattr(self, f).copy_(t)
            else:
                setattr(self, f, t)

    def valid_mask(self) -> Tensor:
        """[B, M] bool — True where a slot has been written.

        The prefix form (`idx < n_stored`) assumes slots fill in order, which is
        true of the ring buffer and FALSE once episodes evict: allocation then
        writes wherever the policy frees room, leaving holes. Ownership is the
        honest test — the plan flagged this exact trap ("valid_mask must become
        ep_id >= 0 ... eviction makes holes") and r31 measured it as 88.4% vs
        100% on a store that never even overflowed.
        """
        if self.cfg.episode_evict:
            return self.ep_id >= 0
        idx = torch.arange(self.keys.shape[1], device=self.keys.device)
        return idx[None, :] < self.n_stored[:, None]

    def norm_age(self) -> Tensor:
        """[B, M] recency in [0, 1] by global write order: newest slot -> 1.

        Used by the recency read to break ties between two slots that match the
        same key token. Free/unwritten slots (slot_seq < 0) map to 0; they are
        masked out of the read anyway, so the value is immaterial there.
        """
        denom = (self.write_count - 1).clamp_min(1).to(self.keys.dtype)[:, None]
        age = self.slot_seq.clamp_min(0).to(self.keys.dtype) / denom
        return age * (self.slot_seq >= 0).to(self.keys.dtype)

    @torch.no_grad()
    def write_many(self, k: Tensor, v: Tensor, pos: Tensor | None = None,
                   tok: Tensor | None = None) -> None:
        """Commit S slots per example — one per token position.

        k, v: [B, S, d_mem]. Slots are taken consecutively from the ring
        pointer. Out-of-place, for the same autograd reason as write().
        """
        B, S, _ = k.shape
        idx = None
        if getattr(self.cfg, "store_grow", False) and not self.cfg.episode_evict:
            self._grow(S)                              # ensure capacity before this write
        M = self.keys.shape[1]                         # live capacity (grows if store_grow)
        S = min(S, M)
        k, v = k[:, :S], v[:, :S]
        idx = self._alloc(S)                       # ring, or episode-aware (r31)
        if not self.training:
            # r108 eval fast path (guarded on eval MODE, not grad-enabled: write_head.commit
            # is @torch.no_grad even in training, so is_grad_enabled would wrongly fire here
            # and corrupt the read's backward graph). No autograd in eval, so write IN PLACE
            # in O(S), not the
            # out-of-place one-hot rewrite of the whole [B,M,d] store (O(M*d) per
            # window -> O(nseg*M*d), which makes a >~10^5-slot store intractable to
            # build). Stored content is detached anyway, so this is byte-identical.
            bidx = torch.arange(B, device=idx.device)[:, None].expand(-1, S)
            self.keys[bidx, idx] = k.detach().to(self.keys.dtype)
            self.values[bidx, idx] = v.detach().to(self.values.dtype)
            self.ptr = (self.ptr + S) % M
            self.n_stored = torch.clamp(self.n_stored + S, max=M)
            self.ep_id[bidx, idx] = self.ep_next[:, None].expand(-1, S)
            if pos is not None:
                self.slot_pos[bidx, idx] = pos[:, :S]
            if tok is not None:
                self.key_tok[bidx, idx] = tok[:, :S]
            seq = self.write_count[:, None] + torch.arange(S, device=idx.device)[None, :]
            self.slot_seq[bidx, idx] = seq
            self.write_count = self.write_count + S
            if self.cfg.center_keys:
                batch_mean = k.detach().reshape(-1, k.shape[-1]).mean(0)
                if bool(self.mean_ready):
                    self.key_mean.mul_(self.cfg.center_mom).add_(
                        batch_mean, alpha=1 - self.cfg.center_mom)
                else:
                    self.key_mean.copy_(batch_mean); self.mean_ready.fill_(True)
            return
        onehot = F.one_hot(idx, M).to(k.dtype)                   # [B, S, M]
        occupied = onehot.sum(1).clamp(max=1)[..., None]         # [B, M, 1]
        self.keys = self.keys * (1 - occupied) + torch.einsum(
            "bsm,bsd->bmd", onehot, k.detach())
        self.values = self.values * (1 - occupied) + torch.einsum(
            "bsm,bsd->bmd", onehot, v.detach())
        self.ptr = (self.ptr + S) % M
        self.n_stored = torch.clamp(self.n_stored + S, max=M)
        # record which episode owns the slots just written
        self.ep_id = self.ep_id.scatter(1, idx, self.ep_next[:, None].expand(-1, S))
        if pos is not None:
            self.slot_pos = self.slot_pos.scatter(1, idx, pos[:, :S])
        if tok is not None:                            # r64: key token per slot
            self.key_tok = self.key_tok.scatter(1, idx, tok[:, :S])
        # r67: stamp a global, monotonic write order on the slots just written, so
        # a later restatement of the same key outranks the earlier one at read.
        seq = self.write_count[:, None] + torch.arange(S, device=idx.device)[None, :]
        self.slot_seq = self.slot_seq.scatter(1, idx, seq)
        self.write_count = self.write_count + S
        if self.cfg.center_keys:
            batch_mean = k.detach().reshape(-1, k.shape[-1]).mean(0)
            if bool(self.mean_ready):
                m = self.cfg.center_mom
                self.key_mean.mul_(m).add_(batch_mean, alpha=1 - m)
            else:
                self.key_mean.copy_(batch_mean)
                self.mean_ready.fill_(True)

    @torch.no_grad()
    def write_gated(self, k: Tensor, v: Tensor, tok: Tensor, vtok: Tensor) -> None:
        """r76: gated memory ops. One slot per DISTINCT key token.

        For each candidate position (in order, so within-segment repeats resolve to
        the latest): if the key token is already stored -> UPDATE that slot's value
        in place (overwrite; a same-value write is a no-op SKIP); else WRITE-NEW at
        the ring pointer. The store therefore holds O(distinct keys) = O(facts), not
        O(tokens), and revisions land on the existing slot. Out-of-place throughout,
        for the same autograd reason as write()/write_many().
        """
        B, S, D = k.shape
        M = self.cfg.memory_size
        dev = k.device
        bidx = torch.arange(B, device=dev)
        arangeM = torch.arange(M, device=dev)
        # Clone the store ONCE (fresh tensors, outside any autograd graph — the
        # current segment's read graph still references the originals), then do cheap
        # in-place index writes per position on the clones and rebind at the end.
        # This turns 768 full-tensor rebuilds into one clone + light scatters.
        keys, values = self.keys.clone(), self.values.clone()
        key_tok, value_tok = self.key_tok.clone(), self.value_tok.clone()
        slot_seq, ep_id = self.slot_seq.clone(), self.ep_id.clone()
        ptr, n_stored, wc = self.ptr.clone(), self.n_stored.clone(), self.write_count.clone()
        for s in range(S):
            ks_, vs_ = k[:, s].detach(), v[:, s].detach()          # [B, D]
            ts_, vts_ = tok[:, s], vtok[:, s]                      # [B]
            if self.cfg.episode_evict:
                valid = ep_id >= 0
            else:
                valid = arangeM[None, :] < n_stored[:, None]       # [B, M]
            match = valid & (key_tok == ts_[:, None])              # [B, M]
            has = match.any(1)                                     # [B]
            mslot = match.float().argmax(1)                        # [B] first match
            idx = torch.where(has, mslot, ptr % M)                 # update slot or new
            # r84: skip pad-valued (or pad-keyed) writes — pad is not a fact, and the
            # spurious value->pad slot is what clobbers a key-and-value token (r83).
            if getattr(self.cfg, "write_skip_pad", False) and self.cfg.pad_id >= 0:
                w = ((vts_ != self.cfg.pad_id) & (ts_ != self.cfg.pad_id)).float()[:, None]
            else:
                w = torch.ones(ks_.shape[0], 1, device=dev, dtype=ks_.dtype)
            keys[bidx, idx] = w * ks_ + (1 - w) * keys[bidx, idx]  # write only where w
            values[bidx, idx] = w * vs_ + (1 - w) * values[bidx, idx]
            wl = w.squeeze(1).long()
            key_tok[bidx, idx] = wl * ts_ + (1 - wl) * key_tok[bidx, idx]
            value_tok[bidx, idx] = wl * vts_ + (1 - wl) * value_tok[bidx, idx]
            slot_seq[bidx, idx] = wl * wc + (1 - wl) * slot_seq[bidx, idx]
            ep_id[bidx, idx] = wl * self.ep_next + (1 - wl) * ep_id[bidx, idx]
            wc = wc + 1
            adv = ((~has).long()) * wl                             # advance only on a real NEW write
            ptr = (ptr + adv) % M
            n_stored = torch.clamp(n_stored + adv, max=M)
        self.keys, self.values = keys, values
        self.key_tok, self.value_tok = key_tok, value_tok
        self.slot_seq, self.ep_id = slot_seq, ep_id
        self.ptr, self.n_stored, self.write_count = ptr, n_stored, wc

    @torch.no_grad()
    def write(self, k: Tensor, v: Tensor, write_mask: Tensor) -> None:
        """Commit one summary per example where `write_mask` is True.

        k, v:       [B, d_mem]
        write_mask: [B] bool
        """
        # OUT-OF-PLACE on purpose. Segments are backpropagated together, so the
        # graph built while reading in segment t still references the store
        # tensors; mutating them in place raises "a variable needed for gradient
        # computation has been modified by an inplace operation". Rebinding
        # instead leaves the earlier tensor intact, which is also the correct
        # semantics — segment t's read really did see the older store.
        M = self.cfg.memory_size
        onehot = F.one_hot(self.ptr, M).to(k.dtype)              # [B, M]
        sel = (onehot * write_mask.to(k.dtype)[:, None])[..., None]   # [B, M, 1]
        self.keys = self.keys * (1 - sel) + sel * k.detach()[:, None, :]
        self.values = self.values * (1 - sel) + sel * v.detach()[:, None, :]
        adv = write_mask.long()
        self.ptr = (self.ptr + adv) % self.cfg.memory_size
        self.n_stored = torch.clamp(self.n_stored + adv, max=self.cfg.memory_size)


# ------------------------------------------------------------------ DG ----
class DentateGyrus(nn.Module):
    """Pattern separation. Ported from the prototype's dg.py::separate.

    Doubles as the query projection: proj maps d_model -> d_mem, so this IS
    W_q and there is no separate matrix.
    """

    def __init__(self, d_in: int, d_mem: int, sparsity: float = 0.1,
                 bypass: bool = False):
        super().__init__()
        self.proj = nn.Linear(d_in, d_mem, bias=False)
        self.sparsity = sparsity
        self.bypass = bypass

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        if self.bypass:
            return F.normalize(h, dim=-1)
        # k >= 2, not 1. With a single surviving unit the code is +-e_i and
        # normalize() returns sign(h_i), which is piecewise constant -> exactly
        # zero gradient to proj, so the query projection silently never learns.
        # Caught by tests_mech at d_mem=16, sparsity=0.1.
        k = max(2, int(self.sparsity * h.shape[-1]))
        topk_vals, _ = torch.topk(h.abs(), k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        return F.normalize(h * (h.abs() >= threshold).to(h.dtype), dim=-1)


# ----------------------------------------------------------- retrieval ----
def settle(q: Tensor, keys: Tensor, values: Tensor, mask: Tensor,
           cfg: MemConfig, stash: dict | None = None,
           pool_mask: Tensor | None = None,
           slot_age: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Batched Hopfield settling over the store.

    q      [B, S, d_mem]   queries (already DG-separated and normalized)
    keys   [B, M, d_mem]
    values [B, M, d_mem]
    mask   [B, M] bool
    returns retrieved [B, S, d_mem], best similarity [B, S]

    The three readout modes are the "hard-max or not" question, kept as one code
    path so they are an ablation rather than three implementations:
      graded — annealed continuous settling, no argmax, fully differentiable
      ste    — hard argmax forward, soft gradient backward (the prototype's)
      topk   — blend the top-k candidates; lets the reader choose among several
               rather than committing to one outside the model
    """
    any_valid = mask.any(dim=-1)                       # [B]
    safe = mask.clone()
    safe[~any_valid, 0] = True                         # avoid an all -inf softmax row

    q0 = F.normalize(q, dim=-1)
    q = q0
    K = F.normalize(keys, dim=-1)
    neg = torch.finfo(q.dtype).min
    graded = cfg.readout == "graded"
    invalid = ~safe[:, None, :]
    if pool_mask is not None:
        # CSA2 candidate pool (r64): score only the pooled slots per query
        # position. Everything outside the pool is treated exactly like an
        # invalid slot, so the read attends over P candidates, not all M.
        invalid = invalid | ~pool_mask

    def weights(cos: Tensor, beta_fwd: float) -> Tensor:
        """Sharp forward, soft backward.

        Sharp retrieval and gradient flow are in direct conflict: measured on a
        4-slot store, d(retrieved)/dq has magnitude ~6 at beta=4, 2e-3 at
        beta=16 and 7e-16 at beta=64 — annealing 8 -> 16 -> 32 -> 64 kills the
        gradient by the second settling step, so the query projection would
        never learn. Resolved the same way CA3 resolves hard argmax: take the
        forward value from the sharp softmax and the gradient from a soft one.
        The returned tensor equals w_sharp exactly.
        """
        s = cos.masked_fill(invalid, neg)
        if slot_age is not None:
            # r67: bias toward the slot written LAST among equal matches, so the
            # newest value of a re-stated key wins. Added AFTER the invalid fill so
            # masked slots stay at neg; small tau keeps a non-match from ever
            # outranking a real match (their cos gap is ~1.0).
            s = s + cfg.recency_tau * slot_age[:, None, :]
        w_sharp = torch.nan_to_num(torch.softmax(beta_fwd * s, dim=-1))
        if cfg.beta_grad is None or cfg.beta_grad >= beta_fwd:
            return w_sharp
        w_soft = torch.nan_to_num(torch.softmax(cfg.beta_grad * s, dim=-1))
        return w_sharp + (w_soft - w_soft.detach())

    beta = cfg.beta
    # n_steps=0 must mean NO settling at all — pure one-shot nearest neighbour —
    # or the ablation cannot distinguish "attractor dynamics matter" from
    # "two iterations instead of five". The graded mode's extra settling steps
    # are therefore only added when settling is enabled at all.
    n_settle = cfg.n_steps + (2 if graded and cfg.n_steps > 0 else 0)
    for _ in range(n_settle):
        w = weights(torch.einsum("bsd,bmd->bsm", q, K), beta)
        # The input stays CLAMPED across settling, as in classical Hopfield.
        # Without it q becomes a function of the keys alone after step 1 and the
        # only route back to q0 is through the (saturating) softmax.
        q = torch.nan_to_num(
            F.normalize(q0 + torch.einsum("bsm,bmd->bsd", w, K), dim=-1))
        if graded:
            beta = min(beta * cfg.anneal, cfg.beta_max)

    w = weights(torch.einsum("bsd,bmd->bsm", q, K), beta)        # [B, S, M]
    if stash is not None:
        # Diagnostic only: which slots the read actually landed on. Lets a recall
        # miss be attributed to MATCHING (wrong slot) or to CONTENT (right slot,
        # value did not survive to the output) instead of guessed at.
        stash["w"] = w.detach()

    soft = torch.einsum("bsm,bmd->bsd", w, values)
    if cfg.readout == "graded":
        retrieved = soft
    elif cfg.readout == "ste":
        idx = w.argmax(dim=-1)                                   # [B, S]
        hard = torch.gather(values, 1, idx[..., None].expand(-1, -1, values.shape[-1]))
        retrieved = hard + (soft - soft.detach())                # STE
    elif cfg.readout == "topk":
        k = min(cfg.top_k, values.shape[1])
        tw, ti = w.topk(k, dim=-1)                               # [B, S, k]
        tw = tw / tw.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        gathered = torch.gather(
            values[:, None].expand(-1, w.shape[1], -1, -1), 2,
            ti[..., None].expand(-1, -1, -1, values.shape[-1]))  # [B, S, k, d]
        retrieved = torch.einsum("bsk,bskd->bsd", tw, gathered)
    else:
        raise ValueError(f"unknown readout {cfg.readout!r}")

    sim = (w * torch.einsum("bsd,bmd->bsm", F.normalize(q, dim=-1), K)).sum(-1)

    # An empty store must yield exact zeros, not NaN — otherwise the zero-init
    # gate below is 0 * NaN and the layer is not a no-op.
    live = any_valid[:, None, None].to(retrieved.dtype)
    return retrieved * live, sim * any_valid[:, None].to(sim.dtype)


# --------------------------------------------------------------- heads ----
class ReadHead(nn.Module):
    """Reads the store into the residual stream at one block."""

    def __init__(self, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        self.dg = DentateGyrus(cfg.d_model, cfg.d_mem, cfg.dg_sparsity, cfg.dg_bypass)
        if cfg.query_init == "identity":
            if cfg.d_mem != cfg.d_model:
                raise ValueError("query_init='identity' needs d_mem == d_model")
            with torch.no_grad():
                self.dg.proj.weight.copy_(torch.eye(cfg.d_model))
            self.dg.proj.weight._keep_init = True   # survives TinyGPT._init
        elif cfg.query_init == "key_proj":
            # The narrow-store analogue of the identity query. With
            # write_identity=False the key is `to_key(h)` for a FROZEN random
            # to_key, so an identity query no longer lands in the key space —
            # and d_mem != d_model makes identity impossible anyway. Copying
            # to_key into the query puts both sides in the same space by the
            # same argument r9 used: query and key must be comparable, and a
            # learned query gets steered away from facts (r8).
            # HippoMemory fills this in once the write head exists.
            self.dg.proj.weight._keep_init = True
            self._needs_key_proj = True
        elif cfg.query_init != "random":
            raise ValueError(f"unknown query_init {cfg.query_init!r}")
        if cfg.freeze_query:
            self.dg.proj.weight.requires_grad_(False)
        self.out = nn.Linear(cfg.d_mem, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model * 2, 1)          # CA1 fusion gate
        # ReZero: alpha=0 makes the layer an EXACT no-op at init, so the model
        # trains as a plain transformer and grows memory use only where it pays.
        # Exactness matters — it is what lets the HC-off baseline be a true
        # control rather than an approximate one.
        self.alpha = nn.Parameter(torch.zeros(1))
        self.stash: dict | None = None      # set to {} to capture retrieval weights
        self._last_r: Tensor | None = None  # r68: raw retrieval (d_mem), pre-out,
                                            # so a later hop can query with it

    def retrieve(self, h: Tensor, store: EpisodicStore, q_src: Tensor | None = None,
                 pool_mask: Tensor | None = None) -> Tensor:
        """Settle the store into a retrieval [B, S, d_mem] — no injection. Split out
        so the loop-read (r82) can settle repeatedly and inject only once."""
        q = self.dg(h if q_src is None else q_src)
        K, V, mask = store.keys, store.values, store.valid_mask()
        if self.cfg.shuffle_store:      # control: example b reads example b-1's store
            K, V, mask = K.roll(1, 0), V.roll(1, 0), mask.roll(1, 0)
        if self.cfg.center_keys:
            # Centre BOTH sides by the same running mean, at match time rather
            # than at write time, so an EMA that drifts cannot leave older slots
            # centred against a stale mean.
            K = K - store.key_mean
            q = q - store.key_mean
        slot_age = store.norm_age() if self.cfg.recency_read else None
        r_raw, _ = settle(q, K, V, mask, self.cfg, stash=self.stash, pool_mask=pool_mask,
                          slot_age=slot_age)
        return r_raw

    def inject(self, h: Tensor, r_raw: Tensor) -> Tensor:
        """Fuse a retrieval into the residual with the CA1 gate + ReZero alpha."""
        r = self.out(r_raw)
        g = torch.sigmoid(self.gate(torch.cat([h, r], dim=-1)))
        inject = self.alpha * g * r
        if self.cfg.read_grad_floor > 0:
            # Zero in value, so the forward — and the exact no-op at alpha=0 — is
            # untouched; in backward the read path sees (alpha + floor) * g.
            floor = self.cfg.read_grad_floor * g * r
            inject = inject + (floor - floor.detach())
        return h + inject

    def forward(self, h: Tensor, store: EpisodicStore, q_src: Tensor | None = None,
                pool_mask: Tensor | None = None) -> Tensor:
        """q_src, if given, is what the query is computed from instead of h (e.g.
        the token embedding without position). The injection still lands on h.
        pool_mask [B, S, M], if given, restricts the read to a candidate pool."""
        r_raw = self.retrieve(h, store, q_src=q_src, pool_mask=pool_mask)
        # r68: expose for multi-hop chaining. r74: keep the graph when hop_grad, so
        # the chain trains end-to-end; otherwise detach (each hop trained alone).
        self._last_r = r_raw if getattr(self.cfg, "hop_grad", False) else r_raw.detach()
        return self.inject(h, r_raw)


class WriteHead(nn.Module):
    """Summarises a segment and offers it to the store."""

    def __init__(self, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        self.to_key = nn.Linear(cfg.d_model, cfg.d_mem, bias=False)
        self.to_value = nn.Linear(cfg.d_model, cfg.d_mem, bias=False)
        # FROZEN, and this is a real limitation rather than an oversight.
        # Only detached copies of k/v ever enter the store, so no downstream loss
        # can reach these projections — left trainable they would sit at their
        # init forever while appearing to be learned. Freezing makes that
        # explicit and gives it a justification: a random projection preserves
        # angles (Johnson-Lindenstrauss), which is all the key space needs, and
        # the learned query projection (DG) can align itself to a fixed target.
        # Learning them needs an auxiliary write objective — v2, alongside the
        # learned write gate. See the plan's Risks section.
        if cfg.freeze_write_proj:
            self.to_key.weight.requires_grad_(False)
            self.to_value.weight.requires_grad_(False)

    @staticmethod
    def _select(hk: Tensor, hv: Tensor, cfg, score: Tensor | None,
                force: Tensor | None = None):
        """Pick which positions earn a slot. Returns (hk, hv) subsampled.

        Every example keeps the SAME count, so the ring pointer stays batched
        and `write_many` is reused unchanged.
        """
        sel = getattr(cfg, "write_select", "all")
        S = hk.shape[1]
        if force is not None:
            # replay an episode's ORIGINAL selection (r32): the policy depends on
            # the weights, so after sleep the same text would be scored differently
            #
            # Bounds are checked rather than trusted. A forced position comes from
            # a DIFFERENT pass over the text, so nothing guarantees it indexes this
            # one -- a short remainder chunk is enough to put it past the end. As a
            # raw gather that is a device-side assert reported asynchronously at
            # the next sync point, which is how it presented in r33: a traceback
            # pointing at an unrelated line in write_many.
            if force.dtype != torch.long:
                force = force.long()
            if int(force.min()) < 0 or int(force.max()) >= S:
                raise IndexError(
                    f"forced write positions must lie in [0, {S}); got "
                    f"[{int(force.min())}, {int(force.max())}]. The episode was "
                    f"written over a longer segment than this one.")
            g = force[..., None].expand(-1, -1, hk.shape[-1])
            return hk.gather(1, g), hv.gather(1, g), force
        if sel == "all" or S == 0:
            allpos = torch.arange(S, device=hk.device)[None, :].expand(hk.shape[0], -1)
            return hk, hv, allpos
        if sel == "last":                       # one slot per segment: the
            idx = torch.tensor([S - 1], device=hk.device)   # v0.1-style summary
            return hk[:, idx], hv[:, idx], idx[None, :].expand(hk.shape[0], -1)
        if sel == "stride":
            k = max(int(getattr(cfg, "write_stride", 1)), 1)
            idx = torch.arange(0, S, k, device=hk.device)
            return hk[:, idx], hv[:, idx], idx[None, :].expand(hk.shape[0], -1)
        if sel == "surprisal":
            frac = float(getattr(cfg, "write_frac", 1.0))
            n = max(int(round(frac * S)), 1)
            if score is None:                   # no score supplied: fall back to
                idx = torch.arange(0, S, max(S // n, 1), device=hk.device)[:n]
                return (hk[:, idx], hv[:, idx],  # stride of the same rate
                        idx[None, :].expand(hk.shape[0], -1))
            sc = score[:, :S]
            idx = sc.topk(n, dim=1).indices.sort(dim=1).values      # [B, n]
            g = idx[..., None].expand(-1, -1, hk.shape[-1])
            return hk.gather(1, g), hv.gather(1, g), idx
        raise ValueError(f"unknown write_select {sel!r}")

    @torch.no_grad()
    def commit(self, h: Tensor, store: EpisodicStore,
               pad_mask: Tensor | None = None,
               h_key: Tensor | None = None,
               score: Tensor | None = None,
               force_idx: Tensor | None = None,
               tok: Tensor | None = None) -> Tensor:
        """Pool the segment, gate on novelty, write. Returns novelty [B].

        h        [B, S, d_model] — hidden states at the write block
        pad_mask [B, S] bool, True on real tokens
        h_key    [B, S, d_model] — states to key on, if not the write block's
        """
        if self.cfg.write_per_token:
            # One slot per token, so the store holds exact content rather than a
            # topic vector, and SHIFTED so a matched key returns what follows it.
            # The right shift depends on what the value layer represents: a
            # mid-stack state is "what is here" (shift 1 reaches the answer), the
            # last block is "what comes next" (shift 0 already is the answer).
            sh = self.cfg.value_shift
            src = h if h_key is None else h_key
            hk = src[:, :src.shape[1] - sh] if sh else src
            hv = h[:, sh:] if sh else h
            hk, hv, pos = self._select(hk, hv, self.cfg, score, force_idx)
            tok_sel = vtok_sel = None
            if tok is not None:                       # r64: key token per slot
                tok_key = tok[:, :tok.shape[1] - sh] if sh else tok
                tok_sel = tok_key.gather(1, pos)
                # r76: the value token = the token the value represents (position
                # shifted by value_shift), so the gate can distinguish update/skip.
                vtok_key = tok[:, sh:] if sh else tok
                vtok_sel = vtok_key.gather(1, pos)
            k_write = hk if self.cfg.write_identity else self.to_key(hk)
            v_write = hv if self.cfg.write_identity else self.to_value(hv)
            if getattr(self.cfg, "write_op_gate", False) and tok_sel is not None:
                store.write_gated(k_write, v_write, tok_sel, vtok_sel)   # r76: new/update/skip
            else:
                store.write_many(k_write, v_write, pos, tok=tok_sel)
            return torch.ones(h.shape[0], device=h.device, dtype=h.dtype)

        def pool(t: Tensor) -> Tensor:
            if pad_mask is None:
                return t.mean(dim=1)
            m = pad_mask[..., None].to(t.dtype)
            return (t * m).sum(1) / m.sum(1).clamp_min(1e-6)

        s = pool(h)
        k = self.to_key(s if h_key is None else pool(h_key))
        v = self.to_value(s)

        mask = store.valid_mask()
        if mask.any():
            sims = torch.einsum("bd,bmd->bm", F.normalize(k, dim=-1),
                                F.normalize(store.keys, dim=-1))
            sims = sims.masked_fill(~mask, -1.0)
            novelty = 1.0 - sims.max(dim=-1).values.clamp(-1.0, 1.0)
        else:
            novelty = torch.ones(k.shape[0], device=k.device, dtype=k.dtype)

        if self.cfg.write_gate:
            write_mask = novelty > self.cfg.write_threshold
        else:
            write_mask = torch.ones_like(novelty, dtype=torch.bool)

        store.write(k, v, write_mask)
        return novelty


class HippoMemory(nn.Module):
    """One store, one write head, and a read head per configured read block."""

    def __init__(self, cfg: MemConfig):
        super().__init__()
        self.cfg = cfg
        # r57: a unique store per layer, or the classic single shared store.
        self.per_layer = bool(getattr(cfg, "per_layer_store", False))
        # r103: per-layer read AND write into the ONE shared store. Reuses the shared
        # `else` branch below (single self.store + a read head per read layer), so
        # reads are per-layer against the shared store; commit() loops the write over
        # every read layer into that same store. Must NOT also enable per_layer_store.
        self.shared_pl = bool(getattr(cfg, "shared_per_layer", False))
        self._layers = tuple(cfg.write_layers) if (self.per_layer and cfg.write_layers) \
            else tuple(cfg.read_layers)
        if self.per_layer:
            self.stores = nn.ModuleDict({str(i): EpisodicStore(cfg) for i in self._layers})
            self.store = self.stores[str(self._layers[0])]   # a default handle
            self.read_heads = nn.ModuleDict({str(i): ReadHead(cfg) for i in self._layers})
        else:
            self.store = EpisodicStore(cfg)
            self.read_heads = nn.ModuleDict(
                {str(i): ReadHead(cfg) for i in cfg.read_layers})
        # r109: shard facts across N shared stores by key token. Shard 0 is self.store;
        # N-1 more alongside. Routing is fixed (tok mod N), read heads are shared.
        self.n_shards = int(getattr(cfg, "n_shards", 1))
        self.sharded = self.n_shards > 1 and not self.per_layer
        if self.sharded:
            self.shards = nn.ModuleList(
                [self.store] + [EpisodicStore(cfg) for _ in range(self.n_shards - 1)])
        self.write_head = WriteHead(cfg)
        # r73: learned reindex — re-lands a retrieved value onto the key manifold so
        # it can address the next hop. One shared map across hops. Init at identity
        # so it starts as the plain retrieved-value-as-key and learns to correct it.
        if getattr(cfg, "hop_reindex", False):
            self.reindex = nn.Linear(cfg.d_mem, cfg.d_model, bias=False)
            if cfg.d_mem == cfg.d_model:
                with torch.no_grad():
                    self.reindex.weight.copy_(torch.eye(cfg.d_model))
                self.reindex.weight._keep_init = True   # survive TinyGPT._init
        self.read_enabled = True
        self.pending_key: Tensor | None = None    # set by the model at cfg.key_layer
        self.pending_keys: dict[int, Tensor] = {}   # per-layer keys (per_layer mode)
        self.pending_values: dict[int, Tensor] = {} # per-layer values (post-block)
        self.cur_tok: Tensor | None = None          # r64: this segment's token ids
        self.pool_cache: Tensor | None = None       # r64: cached candidate pool
        self.hop_q: Tensor | None = None            # r68: previous hop's retrieval,
                                                    # the query for the next read layer
        self.tok_emb_ref: Tensor | None = None      # r75: token embedding matrix [V,d]
                                                    # for the pointer decode (set by model)
        # r64: fixed random hyperplanes for the LSH ("ann") candidate pool
        if getattr(cfg, "read_pool", "none") == "ann":
            g = torch.Generator().manual_seed(0)
            self.register_buffer("lsh_planes",
                                 torch.randn(cfg.d_mem, cfg.lsh_bits, generator=g),
                                 persistent=False)
        if cfg.query_init == "key_proj":
            self._share_key_projection()

    def reset(self, batch_size: int, device=None, dtype=None) -> None:
        self.pending_keys, self.pending_values = {}, {}
        self.pool_cache = None
        self.hop_q = None
        if self.per_layer:
            for s in self.stores.values():
                s.reset(batch_size, device, dtype)
        elif self.sharded:
            for s in self.shards:
                s.reset(batch_size, device, dtype)
        else:
            self.store.reset(batch_size, device, dtype)

    def read(self, block_idx: int, h: Tensor, q: Tensor | None = None) -> Tensor:
        key = str(block_idx)          # ModuleDict has no .get()
        if not self.read_enabled or key not in self.read_heads:
            return h
        if self.per_layer:
            # This depth owns its store; the key is the pre-read state here.
            self.pending_keys[block_idx] = (q if q is not None else h).detach()
            st = self.stores[key]
            if self.cfg.suppress_empty_read and not bool((st.n_stored > 0).any()):
                return h
            return self.read_heads[key](h, st, q_src=q)
        if self.shared_pl:
            # r103: per-layer read against the ONE shared store. Capture this layer's
            # pre-read residual as its key candidate (used by commit if key="residual").
            self.pending_keys[block_idx] = h.detach()
            if self.cfg.suppress_empty_read and not bool((self.store.n_stored > 0).any()):
                return h
            if getattr(self.cfg, "shared_pl_key", "token") == "token":
                # every layer queries the shared token key (exact-match address)
                q_used = self.pending_key if self.pending_key is not None else q
            else:                                   # "residual": key evolves per depth
                q_used = q                          # None at deep layers -> the residual
            pm = self._pool_for_read(q_used if q_used is not None else h)
            return self.read_heads[key](h, self.store, q_src=q_used, pool_mask=pm)
        if self.sharded:
            # r109: route each query position to its shard (tok mod N), retrieve from
            # that shard only, combine, inject once. Each shard holds ~1/N of the keys.
            head = self.read_heads[key]
            tok = self.cur_tok                                  # [B, S] token ids
            qshard = (tok.long() % self.n_shards)               # [B, S]
            r = None
            for s in range(self.n_shards):
                st = self.shards[s]
                if self.cfg.suppress_empty_read and not bool((st.n_stored > 0).any()):
                    continue
                r_s = head.retrieve(h, st, q_src=q)             # [B, S, d]
                sel = (qshard == s)[..., None].to(r_s.dtype)    # [B, S, 1]
                r = r_s * sel if r is None else r + r_s * sel
            if r is None:
                return h
            return head.inject(h, r)
        if self.cfg.suppress_empty_read and not bool((self.store.n_stored > 0).any()):
            # Nothing has been written yet, so there is nothing to retrieve —
            # but the gate is open anyway and the read injects whatever an empty
            # store returns. r34 measured the cost on PG-19's segment 0, which
            # always reads an empty store: -0.0081 nats on seed 1, enough to
            # cancel the benefit of all three later segments, and -0.0024 on
            # seed 2. Off by default: every round up to r35 read the empty store
            # and their numbers include that cost.
            return h
        # r82 (route B): loop-gate latent chaining. At the loop layer, ITERATE the
        # read — settle, hard-decode the retrieval to a token, re-query — loop_iters
        # times, then inject only the final retrieval. Each loop is one hop, entirely
        # latent; the hard decode cleans the signal each hop (fix for r75).
        if getattr(self.cfg, "loop_read", False):
            loop_layer = self.cfg.loop_layer
            if loop_layer is None:
                loop_layer = max(self.cfg.read_layers)
            if block_idx == loop_layer:
                head = self.read_heads[key]
                q_cur = q if q is not None else self.pending_key
                r = None
                for it in range(max(int(self.cfg.loop_iters), 1)):
                    pm = self._pool_for_read(q_cur if q_cur is not None else h)
                    r = head.retrieve(h, self.store, q_src=q_cur, pool_mask=pm)
                    if it < self.cfg.loop_iters - 1:
                        q_cur = self._pointer_key(r, hard=True)   # decode intermediate, re-query
                return head.inject(h, r)                          # inject final hop only
            # non-loop layers: skip the read (the loop layer does the chaining)
            return h
        # r68: multi-hop chaining through the ONE shared store. "token" keeps the
        # model-passed query (current behaviour, unchanged). "retrieved" feeds the
        # previous read layer's retrieval as this hop's key. "residual" queries
        # from the evolved residual stream.
        hop = getattr(self.cfg, "hop_key", "token")
        if hop in ("retrieved", "pointer"):
            if self.hop_q is not None:
                if hop == "pointer":
                    # r75: decode the previous retrieval onto the token-embedding
                    # manifold -> a CLEAN near-token key, discarding off-manifold noise
                    q_used = self._pointer_key(self.hop_q)
                else:
                    # r73: optional learned reindex, else the raw retrieval
                    q_used = self.reindex(self.hop_q) if getattr(self.cfg, "hop_reindex",
                                                                 False) else self.hop_q
            else:
                q_used = q if q is not None else self.pending_key
        elif hop == "residual":
            q_used = None                         # force the residual as query
        else:                                     # "token"
            q_used = q
        pool_mask = self._pool_for_read(q_used if q_used is not None else h)
        out = self.read_heads[key](h, self.store, q_src=q_used, pool_mask=pool_mask)
        if hop in ("retrieved", "pointer"):
            self.hop_q = self.read_heads[key]._last_r   # carry to the next hop
        return out

    def _pointer_key(self, r: Tensor, hard: bool = False) -> Tensor:
        """r75/r82: decode a retrieval onto the token-embedding manifold.

        p = softmax(beta·cos(r, E)); key = p·E — a near-token, on-manifold key, so the
        next hop queries with a clean address and the off-manifold half of the raw
        retrieval (which broke r68/r73/r74) is discarded. hard=True (r82) commits to
        the argmax token's embedding with a straight-through estimator (hard forward,
        soft backward): the hard commit per loop cleans the signal each hop — the fix
        for r75's soft-in-forward accumulation. Only this key crosses to the query.
        """
        E = self.tok_emb_ref                                   # [V, d]
        sims = F.normalize(r, dim=-1) @ F.normalize(E, dim=-1).t()   # [B, S, V]
        p = torch.softmax(self.cfg.hop_ptr_beta * sims, dim=-1)
        soft = p @ E                                           # [B, S, d]
        if not hard:
            return soft
        hard_key = E[sims.argmax(dim=-1)]                      # [B, S, d] argmax token
        return hard_key + (soft - soft.detach())              # straight-through

    def _pool_for_read(self, qv: Tensor) -> Tensor | None:
        """CSA2 candidate pool for this read: build it, or reuse the anchor's."""
        if getattr(self.cfg, "read_pool", "none") == "none":
            return None
        if self.cfg.pool_anchor_layer is not None and self.pool_cache is not None:
            return self.pool_cache                      # cross-layer reuse
        pool = self.build_pool(self.store, qv)
        if self.cfg.pool_anchor_layer is not None:
            self.pool_cache = pool
        return pool

    def _lsh(self, x: Tensor) -> Tensor:
        """Random-hyperplane LSH code (integer per vector) — the 'ann' pool."""
        if getattr(self, "lsh_planes", None) is None:               # lazy (eval flip)
            g = torch.Generator().manual_seed(0)
            self.lsh_planes = torch.randn(self.cfg.d_mem, self.cfg.lsh_bits,
                                          generator=g).to(x.device)
        bits = (x @ self.lsh_planes.to(x.dtype)) > 0                 # [..., L]
        w = (2 ** torch.arange(self.cfg.lsh_bits, device=x.device))
        return (bits.long() * w).sum(-1)                            # [...]

    @torch.no_grad()
    def build_pool(self, store: EpisodicStore, qv: Tensor) -> Tensor:
        """[B, S, M] bool — the candidate slots each query position may read.

        token: slots whose stored key-token equals the query token (exact for
        the position-free layer-0 token-key). ann: slots in the query's LSH
        bucket (approximate, for fuzzy keys). Everything else is masked out of
        the read, so it scores a small pool instead of all M slots.
        """
        valid = store.valid_mask()                                  # [B, M]
        if self.cfg.read_pool == "token":
            tok = self.cur_tok                                      # [B, S]
            cand = tok[:, :, None] == store.key_tok[:, None, :]     # [B, S, M]
        elif self.cfg.read_pool == "ann":
            qc = self._lsh(qv)                                      # [B, S]
            kc = self._lsh(store.keys)                              # [B, M]
            cand = qc[:, :, None] == kc[:, None, :]                 # [B, S, M]
        else:
            raise ValueError(self.cfg.read_pool)
        return cand & valid[:, None, :]

    def set_read(self, enabled: bool) -> None:
        """Disable retrieval while leaving writes intact.

        Sleep replays with reads OFF. If the read head stays live during
        consolidation the model can satisfy the target by retrieving it, so the
        weights never have to absorb anything and nothing consolidates — the
        memory becomes a crutch that hides its own failure to transfer. With
        reads off, the post-sleep HC-off probe measures transfer honestly.
        """
        self.read_enabled = enabled

    def commit(self, h: Tensor, pad_mask: Tensor | None = None,
               score: Tensor | None = None,
               force_idx: Tensor | None = None) -> Tensor:
        if self.per_layer:
            # Each layer writes its OWN post-block state into its OWN store, keyed
            # on the pre-read state captured at that layer during the read pass.
            for i in self._layers:
                v = self.pending_values.get(i)
                k = self.pending_keys.get(i)
                if v is None:
                    continue
                self.write_head.commit(v, self.stores[str(i)], pad_mask,
                                       h_key=k, score=score, force_idx=force_idx,
                                       tok=self.cur_tok)   # r76: enable the op gate per layer
            self.pending_keys, self.pending_values = {}, {}
            return None
        if self.sharded:
            # r109: route each token's (key, value) write to its shard (tok mod N), in
            # place (O(writes)). Same single-store write logic (write_identity, value_shift),
            # just partitioned across N stores so each holds ~1/N of the distinct keys.
            sh = self.cfg.value_shift
            src = self.pending_key if self.pending_key is not None else h
            hk = (src[:, :src.shape[1] - sh] if sh else src).detach()
            hv = (h[:, sh:] if sh else h).detach()
            tok = self.cur_tok
            tok_key = tok[:, :tok.shape[1] - sh] if sh else tok
            B, Sw, d = hk.shape
            sid = (tok_key.long() % self.n_shards)
            bidx = torch.arange(B, device=h.device)[:, None].expand(-1, Sw)
            for s in range(self.n_shards):
                st = self.shards[s]
                m = (sid == s)
                if not bool(m.any()):
                    continue
                cnt = m.sum(1)
                rank = torch.cumsum(m.long(), 1) - 1
                M = st.keys.shape[1]
                idx = (st.ptr[:, None] + rank) % M
                # keys/values are read in the settle graph, so write OUT OF PLACE
                # (clone -> modify -> reassign) or the read's backward is corrupted.
                nk = st.keys.clone(); nk[bidx[m], idx[m]] = hk[m].to(nk.dtype); st.keys = nk
                nv = st.values.clone(); nv[bidx[m], idx[m]] = hv[m].to(nv.dtype); st.values = nv
                st.key_tok[bidx[m], idx[m]] = tok_key[m]
                seq = st.write_count[:, None] + rank
                st.slot_seq[bidx[m], idx[m]] = seq[m]
                st.ptr = (st.ptr + cnt) % M
                st.n_stored = torch.clamp(st.n_stored + cnt, max=M)
                st.write_count = st.write_count + cnt
                if self.cfg.center_keys:
                    bm = hk[m].reshape(-1, d).mean(0)
                    if bool(st.mean_ready):
                        st.key_mean.mul_(self.cfg.center_mom).add_(
                            bm, alpha=1 - self.cfg.center_mom)
                    else:
                        st.key_mean.copy_(bm); st.mean_ready.fill_(True)
            self.pending_key = None
            return None
        if self.shared_pl:
            # r103: write EVERY read layer's own value into the ONE shared store.
            # token mode -> all layers keyed by the shared layer-0 token key (L slots
            # per token, same key, different value); residual mode -> each layer keyed
            # by its own pre-read residual (captured in read()).
            tokmode = getattr(self.cfg, "shared_pl_key", "token") == "token"
            tok_key = self.pending_key.detach() if self.pending_key is not None else None
            for i in self._layers:
                v = self.pending_values.get(i)
                if v is None:
                    continue
                k = tok_key if tokmode else self.pending_keys.get(i)
                self.write_head.commit(v, self.store, pad_mask, h_key=k,
                                       score=score, force_idx=force_idx,
                                       tok=self.cur_tok)
            self.pending_key = None
            self.pending_keys, self.pending_values = {}, {}
            return None
        h_key = None
        if self.cfg.key_layer is not None:
            if self.pending_key is None:
                raise RuntimeError("key_layer is set but no key state was captured; "
                                   "commit must follow a forward pass")
            h_key = self.pending_key.detach()
        self.pending_key = None       # never let a key outlive its segment
        # `score` ranks positions when write_select="surprisal"; the model
        # stashes it during forward (it needs the logits to compute it)
        return self.write_head.commit(h, self.store, pad_mask, h_key=h_key,
                                      score=score, force_idx=force_idx,
                                      tok=self.cur_tok)

    def _share_key_projection(self) -> None:
        """Point every read head's query at the write head's frozen key projection."""
        w = self.write_head.to_key.weight
        for head in self.read_heads.values():
            if getattr(head, "_needs_key_proj", False):
                if head.dg.proj.weight.shape != w.shape:
                    raise ValueError(
                        f"query_init='key_proj' needs matching shapes, got "
                        f"{tuple(head.dg.proj.weight.shape)} vs {tuple(w.shape)}")
                with torch.no_grad():
                    head.dg.proj.weight.copy_(w)

    def gate_magnitudes(self) -> dict[int, float]:
        """|alpha| per read block — the model's own answer to where memory belongs."""
        return {int(i): float(head.alpha.detach().abs())
                for i, head in self.read_heads.items()}

    def mean_occupancy(self) -> float:
        """Mean slots used per example (averaged over per-layer stores). r76: with
        the op gate this stays ~O(distinct keys); without it, it fills to M."""
        if self.per_layer:
            vals = [float(s.n_stored.float().mean()) for s in self.stores.values()]
            return sum(vals) / max(len(vals), 1)
        return float(self.store.n_stored.float().mean())
