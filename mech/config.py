"""
config.py — v0.3 trained-mechanism hyperparameters.

Separate from model/config.py, which configures the v0.1/v0.2 frozen-model
artifact. That one is TF-free and exported to TFLite; this one is PyTorch and
trains. They deliberately do not share a config: `model/config.py::dim` matches
the *encoder* (MiniLM, 384), whereas `d_mem` here is the store's own learned
space and is tied to nothing external.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemConfig:
    # --- shapes -----------------------------------------------------------
    d_model: int = 256
    d_mem: int = 128          # the store's own space. Read and write heads sit at
                              # different depths, so their hidden states are not
                              # comparable; both project into this space instead,
                              # exactly as attention projects into W_Q/W_K space.
    memory_size: int = 512    # M — ring-buffer slots, one per written segment
    store_grow: bool = False  # r100: grow the store like a KV cache (append, amortized
                              # doubling) instead of a fixed ring that laps. Removes the
                              # r51 cliff (slots==M overwrite); memory_size is then just
                              # the initial capacity. Plain write_many path only.

    # --- dentate gyrus ----------------------------------------------------
    dg_sparsity: float = 0.1
    dg_bypass: bool = False   # True = normalized projection, no top-k sparsification.
                              # v0.1 measured DG's separation to HURT semantic recall
                              # (margin +0.143 -> +0.070); here the projection is
                              # LEARNED rather than random, so it gets a fair test.

    # --- fine (bigram) key: DG pattern separation for collisions (r89) -----
    # The exact single-token key cannot separate facts that SHARE a key token
    # (r36's 1/n_facts draw): C facts keyed by the same token collide, all C slots
    # match a query equally. DG separates similar-but-different inputs; here the
    # difference is the CONTEXT token. With fine_key the position-free layer-0 key
    # becomes a bigram, emb(token_t) + alpha*emb(token_{t-1}) — coarse token bucket
    # PLUS a fine context term — so (context, key) pairs sharing a key are pulled
    # apart while the query, which has the same (context, key) adjacency, still
    # reproduces the composite exactly (exact-match preserved, addressing r36's
    # "a single-token key cannot address a multi-token key"). alpha weights the
    # fine term; the query reproduces the bigram, so matching stays exact.
    fine_key: bool = False
    fine_key_alpha: float = 1.0

    # --- CA3 retrieval ----------------------------------------------------
    beta: float = 8.0
    n_steps: int = 3
    anneal: float = 2.0       # per-step beta ramp, graded mode only
    beta_max: float = 64.0    # cap on the anneal ramp
    beta_grad: float | None = 1.0
                              # temperature used for the BACKWARD pass only. Sharp
                              # retrieval and gradient flow conflict: measured
                              # d(retrieved)/dq is ~6 at beta=4, 2e-3 at 16, 7e-16 at
                              # 64. Forward keeps the sharp weights; gradient comes
                              # from this soft one. None = no split (saturates).
    read_grad_floor: float = 0.0
                              # Extra gradient into the READ path that does not
                              # change the forward value. The ReZero alpha scales
                              # both the injection and the gradient reaching the
                              # query projection, so while memory is not yet
                              # helping, alpha shrinks, the matching signal
                              # shrinks with it, and memory stays unhelpful —
                              # measured as the 8-fact gate opening to 0.12 and
                              # closing to 0.05. The floor gives the read path the
                              # first-order signal "this retrieval would lower the
                              # loss if injected" at a fixed strength. 0 = off.
    readout: str = "graded"   # "graded" | "ste" | "topk"
    top_k: int = 3            # readout="topk": how many candidates to blend

    # --- CSA2 candidate-pool retrieval (r64) ------------------------------
    # Instead of scoring q.K over ALL M slots, restrict the read to a small pool
    # of candidate slots chosen by an index. Makes the read O(pool), not O(M).
    read_pool: str = "none"   # "none" (dense, current) | "token" | "ann"
    pool_size: int = 16       # P: candidate slots per query position
    pool_anchor_layer: int | None = None  # build the pool at this read layer;
                              # later read layers REUSE it (cross-layer reuse).
                              # None = build fresh at each read layer.
    lsh_bits: int = 12        # read_pool="ann": random-hyperplane LSH code width

    # --- recency read (r67) -----------------------------------------------
    # A re-stated fact writes a NEW ring slot, so the store holds two slots with
    # the same key token and different values. Without recency the reader blends
    # both (no "latest"). recency_read biases the match toward the slot written
    # LAST (highest slot_seq), so the newest value wins while both stay stored.
    recency_read: bool = False
    recency_tau: float = 0.3        # strength of the recency bias added to cos
                                    # sims before the softmax. Small enough that a
                                    # non-matching slot never outranks a matching
                                    # one (exact-match cos gap ~1.0), large enough
                                    # that among two EQUAL matches the newer wins.

    # --- multi-hop read (r68) ---------------------------------------------
    # Chain reads through the ONE shared store: read layer L's result becomes the
    # key for read layer L+1. "token" = every read layer uses the layer-0 token
    # key (current, single-hop). "retrieved" = the retrieved VALUE re-enters as the
    # next hop's query. "residual" = the evolved residual stream is the next key.
    hop_key: str = "token"          # "token" | "retrieved" | "residual" | "pointer"
    hop_ptr_beta: float = 8.0       # r75: sharpness of the pointer decode. hop_key=
                                    # "pointer" decodes a retrieval onto the token-
                                    # embedding manifold — p=softmax(beta·cos(r,E)),
                                    # key=p·E — so the next hop queries with a CLEAN
                                    # (near-token) key, discarding the off-manifold
                                    # noise the raw retrieval carries. Only this key
                                    # crosses to the next query; the early residual is
                                    # never touched (causal, exact-match preserved).
    # --- loop-gate latent chaining (r82, route B) -------------------------
    # A read layer that ITERATES: settle -> hard-decode the retrieval to a token ->
    # re-query, loop_iters times, then inject the final retrieval. Each loop is one
    # hop, entirely latent (query-only: the intermediate queries never touch the
    # residual). The hard decode (argmax token, straight-through) cleans the signal
    # each hop — the fix for r75's soft in-forward pointer. v1 uses fixed loops
    # (supervised); a learned continue/halt gate is the follow-up.
    loop_read: bool = False
    loop_iters: int = 2             # fixed hops (= n_hops for the supervised v1)
    loop_layer: int | None = None   # which read layer loops; None = the last read layer
    hop_grad: bool = False          # r74: keep the graph across hops (don't detach the
                                    # retrieval fed to the next hop), so the final-answer
                                    # loss backpropagates through every hop into the read
                                    # path and reindex. Trains the chain end-to-end.
    hop_reindex: bool = False       # r73: with hop_key="retrieved", apply a LEARNED
                                    # linear map to the retrieved value before it
                                    # becomes the next hop's key. A raw retrieval is
                                    # scale·(norm(last)+norm(embed)) — off the
                                    # token-embedding key manifold, so exact match on
                                    # the next hop fails (r68/r72). The reindex learns
                                    # to re-land it as a valid key (DeepSeek's
                                    # "reindex" / a thalamic relay). Trainable even
                                    # though hop_q is detached: gradient reaches it
                                    # from the CURRENT hop's read.

    # --- gated memory ops (r76) -------------------------------------------
    # A controller that, per candidate write, picks an OPERATION instead of always
    # appending: write-new (novel key -> allocate a slot), update (key already
    # stored, value differs -> overwrite that slot in place), skip (key+value
    # already stored -> nothing). Keyed on the exact token id, so the store holds
    # ONE slot per distinct key -> O(facts) not O(tokens), and revisions land in
    # place. The learned-gate version comes later; v1 is this exact-key rule.
    write_op_gate: bool = False
    write_skip_pad: bool = False    # r84: don't store slots whose value token is pad —
                                    # pad is not a fact, and the spurious value->pad
                                    # slot is what clobbers a token that is both a key
                                    # and a value (the intermediate in a chain, r83).
    pad_id: int = -1                # set from the tokenizer so write_skip_pad can act

    # --- write policy -----------------------------------------------------
    write_gate: bool = True         # False = always-on (the prototype's behaviour)
    write_threshold: float = 0.6    # novelty above which a segment is stored
    write_identity: bool = True     # Store the layer's OWN hidden states as key
                                    # and value (requires d_mem == d_model).
                                    # to_key/to_value can never receive gradient
                                    # (only detached copies enter the store), so
                                    # as learned maps they are frozen random
                                    # projections that lose exact content and
                                    # cannot be shaped to keep it. Storing raw
                                    # states removes the lossy step entirely and
                                    # is what Memorizing Transformers does.
    value_shift: int = 1            # Value comes from position t+shift, key from
                                    # t. Without the shift, matching a key returns
                                    # that position's own content and never what
                                    # FOLLOWS it, so "find 'ih:' then copy the
                                    # next token" is unrepresentable. The
                                    # prototype writes (dg_codes[j],
                                    # raw_embeds[j+1]) for exactly this reason.
    write_per_token: bool = True    # One slot per TOKEN, not one pooled summary
                                    # per segment. A mean-pooled segment is a
                                    # single vector that must carry an exact
                                    # key->value binding, which is the same
                                    # bottleneck v0.2 measured in hidden-state
                                    # fusion ("Mittens" for Miso): topic
                                    # survives, content does not. Per-token
                                    # slots are also what Memorizing
                                    # Transformers stores.
    center_keys: bool = True        # Subtract a running mean before matching.
                                    # Raw hidden states occupy a narrow cone, so
                                    # every slot matches every query about
                                    # equally — measured here as top-5 cosines
                                    # [0.119, 0.118, 0.118, 0.116, 0.116] over a
                                    # 381-slot store, i.e. no signal at all.
                                    # This is the correlated-key cliff v0.1
                                    # measured (raw LM states ~0.997 mean cosine,
                                    # "unusable"); centering fixed it there too.
    center_mom: float = 0.99        # EMA momentum for that running mean
    shuffle_store: bool = False     # CONTROL: read another example's memories.
                                    # Identical parameters and computation, wrong
                                    # contents — so any gap between this and the
                                    # real store is attributable to the stored
                                    # information rather than to the extra
                                    # capacity the read head adds.
    freeze_write_proj: bool = True  # to_key/to_value get no gradient by construction
                                    # (only detached copies enter the store), so they
                                    # are frozen random projections rather than dead
                                    # trainable params. See WriteHead.

    # --- placement --------------------------------------------------------
    # Write late, read early. The late representation is the most semantically
    # complete thing to store; an early read leaves the rest of the stack free to
    # use what came back. Reading at the last block is the hidden-state fusion
    # that v0.2 measured at 40%.
    read_layers: tuple[int, ...] = (1,)
    write_layer: int = 5
    n_shards: int = 1               # r109: shard facts across N shared stores by key
                                    # token (shard = tok mod N). Each shard holds ~1/N
                                    # of the distinct keys, so a per-store capacity
                                    # frontier (r104-107: ~45% at 128 facts in one store)
                                    # should be relieved N-fold. Read routes the query to
                                    # its shard; write routes each token to its shard.
                                    # 1 = the single shared store (default, unchanged).
    per_layer_store: bool = False   # r57: give EACH layer in write_layers its OWN
                                    # store, keyed and valued at that layer (a unique
                                    # memory per depth), instead of one shared store
                                    # addressed by a single layer-0 key. Tests whether
                                    # removing the layer-0 key bottleneck helps.
    write_layers: tuple[int, ...] | None = None   # r56: if set, the stored value
                                    # is the MEAN of the hidden states at these
                                    # layers (write at every layer), instead of a
                                    # single write_layer. Same slot count, so it is
                                    # a fair A/B against the last-block write.
                                    # None keeps the single-layer behaviour.
    # --- per-layer read/write, ONE shared store (r103, new default going fwd) ----
    # The user's reframe: stop doing "last write, first read" (one early read layer,
    # one late write layer). Instead EVERY layer both reads from and writes to the
    # ONE shared store. Distinct from per_layer_store (which gives each depth its OWN
    # store — regular-transformer stuff, dropped): here the store stays SHARED; only
    # the read/write PLACEMENT changes to all-layers. Read is the existing shared path
    # with read_layers = every block; write loops each layer's post-block value into
    # the same store. Key source per layer is ablated:
    #   "token"    = every layer's slot keyed by the layer-0 position-free token key
    #                (shared address; layers differ only in the value they store) —
    #                keeps exact-token match, but L slots per token share a key.
    #   "residual" = each layer keyed by its OWN pre-read residual (transformer-native
    #                per-depth address) — tests whether the evolving stream can key.
    shared_per_layer: bool = False
    shared_pl_key: str = "token"    # "token" | "residual"
    query_init: str = "random"      # "identity" starts the query projection as
                                    # I. With key_layer == the read layer and raw
                                    # stored keys, query and key are then the SAME
                                    # function of the same state at step 0, so an
                                    # identical token matches itself from the
                                    # start. The key side cannot learn (it enters
                                    # the store detached), so the query side is the
                                    # only place alignment can come from; a random
                                    # start makes it find identity through a
                                    # 256x256 map. r2 measured that plateau: right
                                    # key position, wrong key, recall pinned at
                                    # ~1/n_facts. Needs d_mem == d_model.
    value_skip_layer: int | None = None
                                    # Add this block's output to the write-block
                                    # state before it becomes the value: a skip
                                    # connection into the store. The write still
                                    # happens at write_layer, but the value also
                                    # carries an earlier, identity-bearing state.
                                    # r9: a block-0 value stores single-occurrence
                                    # facts at 100%, a last-block value does not.
    value_skip_norm: bool = False   # With value_skip_layer: unit-normalise both halves

    # --- write RATE (r30) -------------------------------------------------
    # Causal attention means position t's state already summarises everything
    # <= t, so one slot per token stores T overlapping summaries of the same
    # episode. These pick WHICH positions earn a slot. The store is then a
    # subsampled transcript, so decode coverage (r26) has to be reported
    # alongside recall — sleep replays from what was written, not from the text.
    # --- episodes (r31) ---------------------------------------------------
    # The store accumulates episodes and must evict under pressure. The rule is
    # the user's: replace stale INCOMPLETE episodes, never a complete one — and
    # completeness is exactly what makes an episode consolidatable by sleep
    # (r26/r27), so this is the same distinction seen from the eviction side.
    episode_evict: bool = False     # False = plain ring buffer (every round <= r30)
    max_episodes: int = 64          # episode-id space; ids wrap, slots do not

    suppress_empty_read: bool = False   # skip the read while the store is empty.
                                    # r34: the gate is open at segment 0 with
                                    # nothing to retrieve, and the model pays
                                    # for it (-0.008 nats on one seed). Off by
                                    # default -- r1-r35 all read the empty store
    write_select: str = "all"       # all | stride | surprisal | last
    write_stride: int = 1           # write_select=stride: every k-th position
    write_frac: float = 1.0         # write_select=surprisal: top fraction by
                                    # the model's own NLL on the token the slot
                                    # would return (r6 inverted: do not store
                                    # what the model can already predict)
                                    # before summing, so the last-block state cannot
                                    # swamp the identity-bearing skip. Each half is
                                    # rescaled to the skip half's mean norm, keeping
                                    # the value on the scale the read path saw
                                    # without normalisation.
    freeze_query: bool = False      # Hold the query projection at its init (use
                                    # with query_init="identity"). One projection
                                    # serves every position, so text positions
                                    # far outnumber the answer position in its
                                    # gradient. r7/r8 measured the result on
                                    # single-occurrence facts: matching 100% ->
                                    # 0%, yet on the collapsed checkpoint an
                                    # UNTRAINED identity query still matches 93.8%
                                    # while the learned one (only 17% from I)
                                    # matches 0%. Training steered it away.
    key_pos_free: bool = False      # With key_layer == read layer == 0, take the
                                    # key AND the query from the token embedding
                                    # alone, without the position embedding. A
                                    # layer-0 state is token + position; r7 saw
                                    # matching fall 99% -> 0% on single-occurrence
                                    # facts, where the facts sit at positions 0-15
                                    # but the query and the distractor text share
                                    # positions near the segment's end — position
                                    # similarity can outvote token identity.
    key_layer: int | None = None    # Where the KEY half of a slot comes from.
                                    # None = the write layer, like the value.
                                    # r1 measured why that fails at the last
                                    # block: it encodes what comes NEXT, not what
                                    # is here, so a layer-1 query that knows only
                                    # token identity cannot match it. Setting this
                                    # to the read layer puts keys in the query's
                                    # own space — attention's arrangement, with
                                    # W_q and W_k at one depth — and leaves only
                                    # the value to come from the last block.
