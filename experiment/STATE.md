# STATE — resume point for the experiment loop

**Mode: AUTONOMOUS.** Cron fires :07/:37 and runs the full loop. Two boxes, different work in parallel.

**Boxes**
- **box1** = `ssh box1` (adhavan_23bce0839@10.30.161.19), `~/research/framework`. Monitor: `monitor.sh rN`.
- **box2 "vanilla"** = <user>@<box2-host>. Password auth via the helper
  `/tmp/claude-1000/-home-tba-projects-framework/fcd349d4-72d9-4988-9c2f-5ceac1569fb6/scratchpad/b2ssh`, which is a
  **drop-in `ssh` wrapper** (`./b2ssh box2 'cmd'`, and `rsync -e ./b2ssh`), not an askpass helper.
  **Work lives in `~/fw`** — `~/research/framework` there belongs to another account. `~/fw` has its own venv and
  **`~/.local/bin/uv` works there** (the older note saying it does not is wrong). **nvidia-smi is broken**
  (driver/library mismatch, NVML 595.91); torch still reports CUDA fine, so check the GPU with torch, not smi.
- **Launch detached or lose the run.** `setsid nohup ... < /dev/null &` with stdout redirected to a log. r33 was first
  launched attached and the 600s ssh client timeout killed it — no log, no result, and it *looked* alive for 622s.
- `uv` is not on the non-interactive ssh PATH: call `~/.local/bin/uv` by absolute path.

## ===== LIVE @19:12 — r103 DONE+WON (4425d93); r104 capacity test RUNNING =====
r103 DONE (per-layer r/w = new default): per-layer WRITE (each layer its own value, 6 slots/tok) + exact TOKEN key
beats read-all/write-MEAN **80.8% -> 95.5%/98.9%** (2 seeds) on cold bpe_recall, bootstraps earlier. CONTROL spl_residual
(same 6x slots, residual key not token) = **0.2%** => win is per-depth VALUES + exact-token addressing, NOT capacity.
RESULTS + README + paper method fold (20pp, builds) + commit 4425d93. Validates user's "per layer r/w" reframe.
r104 RUNNING (does the +16pt win move the r88 ~128-fact capacity WALL?): matched baseline_mean vs spl_token at higher
facts. box1 @128 facts, box2 @64 facts, each 2 arms (baseline ~40min + spl_token ~75min), n_seg 8. Started ~19:11.
r104 BASELINES DONE @20:40: baseline_mean_f128 = **0.3%** (mean-write CANNOT learn 128 facts = r88 wall confirmed),
baseline_mean_f64 = **96.2%** (fine at 64). Both spl_token arms then CRASHED (CUDA OOM).
*** DETACH BUG FOUND+FIXED @20:45 (model.py) ***: shared_pl stored pending_values[i]=x WITH the autograd graph (not
detached like the normal path's commit(h_write.detach())). A later segment's read linked back through every earlier
segment's write -> graph retained across all n_seg segments -> OOM at n_seg 8 (r104 spl crashed). Also may have
INFLATED r103 via accidental cross-segment BPTT. Fixed: pending_values[i]=x.detach() (stored content never gets grad
by design). Smoked OK. => RE-VERIFYING r103 + running r104 capacity on both boxes via experiment/r104/spl_fixed.sh:
  (1) r103_spl_token_detach_s{0,1} (n_seg4, 32 facts) — does +16pt SURVIVE detach? (2) r104_spl_token_f{128,64}.
  box1 seed0/128f, box2 seed1/64f. Started ~20:45/20:48. reverify signal ~21:45, final ~22:10; capacity ~23:40.
*** BOX2 GOTCHA: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CRASHES box2 (nvmlInit assert — box2's NVML is
broken, same as nvidia-smi). box2 CUDA compute is FINE. So expandable_segments is box1-ONLY in spl_fixed.sh; box2
relies on the detach fix alone. NEVER set expandable_segments on box2. ***
IF reverify holds (~95/99%): r103 claim solid, keep paper. IF it drops: r103 win was the BPTT artifact -> amend r103
RESULTS + paper method + README. Next tick: harvest reverify verdict FIRST (protects committed claim 4425d93).
PROGRESS @21:10: both reverify arms ALIVE+advancing (box1 detach_s0 step3000, box2 detach_s1 step1500; box2 slower).
REASSURING: box1 detach_s0 @step3000 values IDENTICAL to original r103 seed0 @step3000 (train 13.0436, same gates) =>
detach barely changes early trajectory (cross-seg grad negligible while gates small) => win likely REAL not BPTT.
But both PRE-BOOTSTRAP (0.3-0.9%); verdict at bootstrap ~step5-7k (~21:45-22:10 box1). GOTCHA: r104_spl_token_f128.log
(box1) + f64.log (box2) are STALE pre-fix CRASH logs (OOM/NVML) until the reverify finishes and the capacity arm
overwrites them — do NOT re-panic at those. Both boxes commit 29a0be0 (detach fix). Fixed script = experiment/r104/spl_fixed.sh.
*** REVERIFY CONFIRMED @22:10: r103 +16pt win is REAL, NOT a BPTT artifact. ***
box1 detach_s0 FINAL = **0.955** — EXACTLY the original r103 seed0 (0.955); detach changes NOTHING (pure memory fix).
box2 detach_s1 step5000 = 87.8% climbing (heading to original 0.989). r103 claim SOLID, paper 4425d93 stands as written.
CAPACITY (r104): box1 f128 OOM'd EVEN with detach+expandable_seg (128 facts x n_seg8 store ~8x reverify = genuinely
~16G). RETRY at BATCH 16 (halves mem, fits) RUNNING @22:12 -> r104_spl_token_f128.json. Compare to baseline_f128=0.3%
(batch caveat: baseline was batch32, but 0.3% vs any high spl = clear regardless). box2 STILL on reverify (slow,
~17min/1k, done ~23:38) then f64 capacity. NEXT: harvest f128 capacity verdict + box2 reverify final + f64;
then write r104 RESULTS (detach-bug story + reverify robustness + capacity verdict) + fold.
PROGRESS @22:40: box2 reverify seed1 = **98.0%** @step7000 (=> SEED1 CONFIRMED, ->orig 0.989). box1 f128-b16 COLD
@step1500 (0%, batch16 slow ~18min/1k -> verdict ~23:50+). WATCH: box2 f64 arm is scripted batch32 & WILL likely OOM
(64f x n_seg8 store, box2 no expandable_seg) -> let it fast-fail, RELAUNCH f64 at BATCH 16 when box2 frees (like box1 f128).
OPEN/NEXT candidates (after r104):
- r105 per-layer write on MULTI-HOP CoT (bpe_chain, r93 recipe + --shared-per-layer): tests if the +16pt single-hop
  win carries to multi-hop. DESIGN TENSION: r93 recipe uses --write-op-gate (1 slot/distinct-token) which COLLAPSES
  per-layer slots (same token x6 layers -> dedup to 1), killing the per-layer benefit. So SPL arm must DROP op-gate
  (plain per-layer write + grow); add a baseline-no-opgate arm to isolate. Ready-ish (no new code) but needs matched design.
- LATENT multi-hop (needs CODE): per-layer reads CHAIN (layer i retrieval -> layer i+1 query via _pointer_key), L
  layers = L latent hops, NO emitted tokens. vs emitted-token CoT (r84/r95). Design+smoke read-chaining first.
- per-layer write on a REAL LM (PG-19 ppl on/off, mean-vs-per-layer) = the honest reviewer benchmark (plan r71).
r93 multi-hop recipe (for reference): --task bpe_chain --n-hops N --n-chains 4 --recency-read --write-op-gate
  --write-skip-pad --value-skip-layer 0 --value-skip-norm --read/write-layers 0-5 --memory-size 1024/2048 --init BASE.
--- (superseded) r102 saga + r103-launch notes below ---
## ===== USER @13:20 "kill r107, go 3 and 2 parallel, kv after, defer 4/5" — r108(#3)+r109(#2) RUNNING =====
r107 KILLED (frontier mapped enough: 64~96/96=62.8/112=52.2/128=45.3, smooth slope; f80/f88 abandoned).
#3 STREAMING 1M EVAL (r108, box1): ROOT CAUSE of the 1M OOM found = write_many rewrites the WHOLE [B,M,d] store
out-of-place via F.one_hot(idx,M) EVERY window (O(M*d)/window -> intractable at ~1M slots). FIX: eval-mode in-place
O(S) write (guard `not self.training`, NOT is_grad_enabled — write_head.commit is @no_grad even in training; using
is_grad_enabled corrupts the training read graph, caused an autograd error, fixed). Now fixed-M + batch1 + top-1 builds
a ~1M-slot store fast. r108: ns1000 100% in 2.5min (r102 took 10min+OOM); ns3000/ns7800(~1M tok) building. commit 3d6bde2.
#2 SEVERAL SHARED STORES (r109, box2): n_shards flag — shard facts by key token (tok mod N), N stores, per-shard
retrieve+combine read, out-of-place sharded write. Tests 1/2/4 shards @128 facts (does sharding relieve the per-store
frontier? = user's addressing/KV framing). batch 8 (box2 GPU contended ~10GB by others). commit 1d2b67d.
NEXT: #1 KV/RoPE positional differentiation AFTER r108+r109. #4 (int4 kernels) + #5 (learned gate) DEFERRED.
HARVEST: r108 -> literal 1M-token number into paper §longcontext (replaces "eval-bound ~128k"); r109 -> does sharding
close 128? into limitation #2.

## ===== @22:05 USER "run o7 gn" — OFFLINE; r112 PASSKEY BUILT + CHAINED on both boxes =====
User authorized the autonomous benchmark push and went to sleep ("run o7 gn buddy"). r112 = PASSKEY (needle-in-a-
haystack), the top benchmark-targets item, is BUILT, SYNCED to both boxes, and CHAINED to fire when the r111 1M cells land.
**r112 hypothesis**: our store is content-addressed by a position-free layer-0 key => passkey recall is INVARIANT to needle
DEPTH (no "lost in the middle", unlike attention). Length axis already owned (r108: 1M). r112 isolates DEPTH.
**Code added (LOCAL, UNCOMMITTED — commit on harvest)**: (1) tasks.py `bpe_recall_batch(needle_seg=0)` places the fact in
window d; default 0 = byte-identical RNG to every prior round (verified by reasoning: same draw order). (2) train.py get_batch
threads `needle_seg`. (3) persist.py NEW `passkey` probe (sweeps --depth-list at fixed ctx, top-1, --shuffle control) +
registered in choices/dispatch + `--depth-list` arg. All 3 ast.parse OK. Synced tasks/train/persist.py + experiment/r112/ to
box1(~/research/framework) AND box2(~/fw). model.py/config.py/hippo_layer.py UNCHANGED (sparse path already works on both).
**r112 run** (experiment/r112/run.sh box1|box2): SMOKE (1 iter, fails fast) -> ctx64(~8k) depth sweep -> ctx512(~64k) depth
sweep -> ctx64 --shuffle control. batch 2, top-1. Fast (~min/cell, recall evals not training). Depths {0,.25,.5,.75,1.0}.
**CHAINS launched detached NOW**: box1 chain PID **45267** waits on rare PID **43196** then runs r112 box1;
box2 chain PID **633688** waits on cross-model PID **631056** then runs r112 box2. (`ps -p <pid>` poll, 120s; ->run.sh.)
@22:05 status: box1 rare_ns7800 RUNNING 1h23 (no result); box2 box2_ns7800 RUNNING 3h14 (no result). Both chains armed.
@22:35 UPDATE — r111 rare LANDED + r112 SMOKE CAUGHT A BUG (fixed) + first passkey numbers:
  * r111 rare_ns7800 DONE = **75.0%** top-1 @ 990,473 slots (rare-key 1M; key-quality effect persists — normal=100%). HARVEST w/ box2.
  * r112 box1 chain fired, SMOKE FAILED FAST (18s, guard worked): TypeError n_shards — box1's config.py was STALE (r103/r109
    never synced to box1; I'd wrongly assumed config/model/hippo unchanged). FIX: rsync'd FULL mech/*.py to BOTH boxes
    (defaults shared_per_layer=False/n_shards=1 preserve r87/r92 checkpoint arch — box2 already ran this way). RELAUNCHED
    r112 box1 direct (idle, rare done). SMOKE OK now. LESSON: sync FULL mech/, not a subset, when train.py's build() changed.
  * r112 box1 ctx64 (8k) depth sweep DONE: depth 0/.25/.5/.75/1.0 = **100/100/100/87.5/75%**. => NO lost-in-the-middle
    (middle=100%), but a MILD RECENCY-END DIP (recent needle slightly worse). n=8 (iters4*batch2) => NOISY (1-2 misses).
    ctx512(64k) running now; box2 cross-model chain still armed. IF the recent-end dip reproduces at ctx512/box2 -> real
    effect worth a higher-iters (n>=32) characterization + a mechanism note; IF not -> noise, headline = flat/no-LitM.
    NOTE for harvest: consider a higher-iters rerun (--iters 16) to de-noise the depth curve before folding into paper.
@22:50 BREAKTHROUGH — the recency dip is REAL and STRONG at scale (r112 ctx512 DONE):
  * r112 box1 ctx512 (64k): depth 0/.25/.5/.75/1.0 = **100/62.5/50/25/12.5%**, shuffle control **0%** at all depths.
    => NO lost-in-the-middle (no MID penalty), but a STRONG **PRIMACY / write-order decay**: fact stated EARLY persists
    (100%), fact stated LATE into a SATURATED store is nearly lost (12.5%). Penalty SCALES with saturation (mild 8k -> severe
    64k). Opposite of a KV cache's recency. Likely the SAME limit as the 128-fact capacity frontier along the time axis.
    r112 box1 fully DONE (ctx64+ctx512+ctrl). box2 cross-model depth sweep still pending via its armed chain.
  * r113 LAUNCHED on box1 (idle after r112) to CHARACTERIZE (behavioral, no mech-interp): (a) saturation sweep ctx 32/128/512
    [ctx32 done = 100/100/87.5/81/81 -> mild at low sat, CONFIRMS saturation-scaling]; (b) --rare-keys @ctx512 (unique
    always-novel keys -> FLATTENS iff the novelty write-gate write_threshold=0.6 is suppressing late writes); (c) --store-grow
    @ctx512 (never-laps control -> should NOT change). iters8/n16 de-noised. Reads: rare-keys flat => gate (fixable policy);
    rare-keys still decays => center_keys drift (representational). ETA ~40min.
  * BUG (fixed): r113 first launch redirected to experiment/r113/logs/_launch.log but rsync doesn't create EMPTY dirs, so
    logs/ was absent -> redirect failed -> nothing ran -> box1 briefly idle. FIX: mkdir -p logs first + redirect launch to /tmp.
    LESSON: before `setsid nohup ... >DIR/x.log`, ensure DIR exists (rsync of a folder won't create its empty subdirs).
  * HARVEST PLAN (next crons): fold r111(rare 75% + xmodel 1M) + r112(write-order decay, both models) + r113(mechanism) into
    ONE story -> paper §longcontext/§passkey + §limitations: "content-addressed memory has NO lost-in-the-middle but a
    write-order/saturation limit; early facts privileged" + connect to capacity frontier. Commit incl. r112/r113 infra code.
    This is a REPORT-WORTHY finding for the user on wake (overturns the naive 'flat' hypothesis with a sharper, honest result).
@23:05 CROSS-MODEL CONFIRMS (box2 r111 landed + r112 box2 running):
  * r111 box2_ns7800 (cross-model 1M, d256) DONE = **100%** top-1 @ 990,473 slots. => LITERAL 1M now on TWO models (d512
    r108=100% + d256=100%). r111 FULLY DONE: {ns1000/3000=100% both models} + {rare 1M=75%} + {xmodel 1M=100%} + {batch-2
    controls 0%}. READY TO HARVEST.
  * r112 box2 (cross-model DEPTH sweep, d256) chain FIRED, smoke OK. box2_ctx64 (8k): depth 0/.25/.5/.75/1 = **87.5/0/0/0/0%**
    => on the WEAKER model the write-order decay is NEAR-BINARY: ONLY the first-written fact survives; lost by window ~16/64.
    (d512 @8k was graded 100/100/100/87.5/75.) So primacy/write-order decay is CROSS-MODEL ROBUST — graded on strong model,
    catastrophic on weak. Monotonic from depth 0 on BOTH => definitively NOT lost-in-the-middle. box2_ctx512 running now.
  * BOTH BOXES BUSY: box1 r113 (mechanism: sat/gate/centering, d512), box2 r112 ctx512 (cross-model, d256). Healthy.
  * r111 HARVESTED @23:10 (this cron): r111/RESULTS.md (1M on 2 models + controls 0% + rare-key 75%@1M), README row, folded
    cross-model-1M + controls into §longcontext, pulled result JSONs local, committed. r111 DONE+banked.
  * STILL PENDING HARVEST (both rounds finishing now): r112 (write-order decay, d512+d256) + r113 (mechanism: sat/gate/
    centering). NEXT CRON: r112/RESULTS + r113/RESULTS + README rows + MAJOR paper update (new §passkey or §longcontext para
    + §limitations: write-order/saturation/primacy limit, cross-model, ties to capacity frontier) + commit r112/r113 INFRA
    code (needle_seg in tasks.py/train.py + passkey probe in persist.py — currently synced to boxes but UNCOMMITTED locally).
    Do NOT re-tune to flat — report the decay as measured. r113 sat sweep so far: ctx32=100/100/87.5/81/81, ctx128=100/87.5/
    68.8/56.2/50, ctx512(fine) running; rarekeys+grow arms after (the gate-vs-centering discriminator).
@23:15 r111 COMMITTED (fb21aab, no trailer). r112 box2 FULLY DONE (ctx64 d256=87.5/0/0/0/0, ctx512 d256≈all-0 past depth0,
  + shuffle control). BOTH BOXES now on r113 mechanism: box1=r113 d512 (sat_ctx512 running -> rarekeys -> grow), box2=r113
  d256 CROSS-MODEL mechanism (chain_r113_box2 fired clean on r112 ROUND DONE; running d256_sat_ctx32 -> ... -> rarekeys/grow).
  Outputs: box1 = experiment/r113/results/{sat_ctx*,rarekeys_ctx512,grow_ctx512}.json; box2 = same dir, d256_ prefixed.
  NEXT CRON HARVEST (both r113 arms + r112 both models): r112/RESULTS + r113/RESULTS + README rows + paper §passkey/
  §limitations (write-order/saturation/primacy, cross-model, gate-vs-centering verdict from rarekeys arm) + commit r112/r113
  INFRA (needle_seg + passkey probe, still uncommitted). KEY READ: does --rare-keys flatten the decay? flat=>novelty write-gate
  (write_threshold=0.6) suppressing late writes (a fixable policy); still-decays=>center_keys drift (representational).

## ===== @00:28 (Sep 23) SESSION WOUND DOWN + CLOSED (user: "wind it down, check and close it up") =====
CRON 34e14d22 DELETED (autonomous loop OFF). Passkey finding fully harvested + committed. Paper builds (pdflatex x3, 27pp).
FINAL RESULT (r112+r113, both models, controls 0%): content-addressed memory has **NO lost-in-the-middle** (recall monotonic
in depth, mid-context never anomalous) but a **WRITE-ORDER / primacy decay** — early fact persists, late fact into a saturated
store is lost (d512@64k: 100/62.5/50/25/12.5% by depth; d256 near-binary 87.5/0/0/0/0). MECHANISM (r113, behavioral): (a)
SATURATION-driven (steepens 81->50->6% as store fills 4k->16k->65k slots); (b) NOT ring lapping (grow==baseline identical);
(c) PARTLY the novelty write-gate write_threshold=0.6 (--rare-keys lifts late tail 6->25%, a FIXABLE policy) + a residual
factor (center_keys drift, hypothesis, no mech-interp). Ties the 1M long-context reach to the 128-fact capacity frontier along
the TIME axis. HARVESTED: r112/RESULTS + r113/RESULTS + 2 README rows + paper §longcontext new "Passkey" paragraph + rare-key
1M note. COMMITTED with infra (needle_seg in tasks.py/train.py + passkey probe + --depth-list in persist.py).
box2 was mid d256_rarekeys_ctx512 at wind-down; it + d256_grow later LANDED and were FOLDED IN (@02:15): cross-model r113 =
not-lapping (grow==baseline 81/0) and saturation (near-binary) REPRODUCE on d256, but the write-gate rescue is MODEL-DEPENDENT
— d256 --rare-keys does NOT lift the tail (31/0/0 vs d512's 6->25) because its rarest embeddings are too under-trained to key.
r113/RESULTS + paper (chronicle/limitations/appendix) updated; box2 now IDLE. All boxes idle, paper current at r113.
OPEN AGENDA when research resumes (all deferred to user): (1) WRITE-GATE FIX round — retrain/eval with a lower/disabled novelty
gate or salient-token key, does it flatten the write-order decay? (natural next step from r113). (2) benchmarks BABILong then
FANToM (need harnesses; passkey DONE). (3) sharding debug (r109 buggy). (4) KV/RoPE differentiation (position vs content
addressing — user's design call). (5) int4/hash kernels + learned salience gate (deferred). NO Co-Authored-By (CLAUDE.md).
**NEXT WAKEUP**: (a) HARVEST r111 when the 1M cells land -> r111/RESULTS (1M on 2 models + controls 0% + rare-key-at-1M),
README row, paper §longcontext, commit (INCLUDES the r112 infra code). (b) Check r112 chain: `_chain_box{1,2}.log`,
`_smoke_box{1,2}.log`, `_progress_box{1,2}.log`. If SMOKE FAIL -> debug passkey probe (my code). (c) HARVEST r112 when
depth sweep done -> if recall FLAT across depths (expected): NEW paper result "no lost-in-the-middle" (fold into §longcontext
or a short §passkey), README, commit. THEN BABILong (needs multi-hop CoT), then FANToM. Sharding-debug + RoPE await user.

## ===== 1M DONE @17:52 (162709f) + r109 sharding BUGGY + r111 launched =====
r108 1M PUSH DONE: top-1 FLAT **100%** at ns1000/3000/7800 = 126,873 / 380,873 / **990,473 slots (~998k≈1M tok)**,
1 slot. Eval bug (O(M*d)/window write) fixed -> 1M MEASURED. HARVESTED: r108/RESULTS, README, paper longcontext+
conclusion (now claim 1M, was eval-bound ~128k), builds 27pp. commit 162709f.
r109 SHARDING (box2) DONE but SUSPECT: shard1/2/4 ALL **0.0%** @128 facts. 4-shard=32 facts/shard SHOULD recall ~96%
(single 32-fact store works) -> my sharding impl is almost certainly BUGGY (read/write routing), NOT a real "sharding
doesn't help". INCONCLUSIVE, NOT folded into paper. Needs debug (my code, borderline; deferred — do not deep-dive now).
r110 (box1 chained sharding@256) FAILED: box1's train.py lacks --n-shards (synced train.py to box2 not box1). Moot
until sharding debugged.
BOX1 2M push OOM'd: d512 read materialises normalize(keys) over M (4GB@2M) + 8GB store > 16GB. So **1M is d512's
ceiling on 16GB** without a chunked read (avoid materialising [B,S,M]/normalize-over-M). 2M+ needs that refactor.
r111 LAUNCHED @17:56: box1 = 1M SCRAMBLED CONTROL (ns3000/7800 --shuffle, expect ~0%, confirms 1M genuine);
box2 = CROSS-MODEL 1M (d256 r92 ckpt, ns1000/3000/7800, confirms 1M generalizes). Both solid, fit, ~2h.
HARVEST next: r111 box1 control (~0% = 1M genuine) + box2 cross-model 1M -> fold both into paper longcontext (1M on
two models + control). THEN benchmarks (passkey first, memory benchmark-targets.md). Sharding debug + RoPE = user's return.
@18:40 GOTCHA: box1's batch-1 scrambled control gave 100% (NOT a real result) — shuffle_store permutes across the BATCH,
so at batch 1 there's no neighbour to permute with -> degenerate (reads own store). The 1M-scale scrambled control CANNOT
use batch 1. Genuineness stands on r102's valid batch-2 controls (0-0.6%) + the mechanism. Killed it; relaunched box1
VALID control = ns1000/ns3000 batch 2 --shuffle (fits; expect ~0%). box2 cross-model 1M: ns1000=100%, ns3000=100% on
d256 (r92) -> 1M generalizes; ns7800 building. NEVER run the shuffle control at batch 1.
@20:10: box1 VALID CONTROLS DONE = ns1000 **0.0%** + ns3000 **0.0%** (batch-2 shuffle) -> long-context retrieval is
GENUINE at 128k AND 384k (scrambled->chance, real->100%). box2 cross-model 1M: ns1000/3000=100% on d256, ns7800 (1M)
still building ~20min. box1 idle -> launched rare-key 1M (ns3000/7800 --rare-keys): does r102's rare-key
underperformance (50-62% @128k) persist/worsen at 1M? HARVEST when box2 ns7800 lands: r111 = 1M on TWO models
(d512 r108 + d256) + valid controls 0% + rare-key-at-1M -> fold into paper longcontext. commit pending.

## ===== OFFLINE @14:00 — user going offline "for a while"; autonomous loop follows THIS queue =====
QUEUE (boxes busy hours; cron loop harvests+advances each wakeup):
- box1: r108 (1M push) RUNNING -> CHAINED to r110 (sharding capacity scaling @256 facts, 2/4/8 shards) via
  ~/research/framework/chain_r110.sh (waits for "1M PUSH DONE"). So box1 = 1M then shard-scaling, no idle.
- box2: r109 (sharding @128: 1/2/4 shards) RUNNING (~6-8h at batch 8, GPU contended).
LOOP TODO while offline (in order):
  1. r108 done -> r108/RESULTS (the LITERAL 1M-token number, best+final) + README + paper §longcontext (replace the
     "measured ~128k, 1M eval-bound" with the real number; the write fast-path made it tractable). commit.
  2. r109 done -> r109/RESULTS: does sharding relieve the 128 frontier (2/4-shard vs 1-shard 45%)? + README + paper
     limitation #2 (if sharding closes it, "several stores" moves from open-direction to demonstrated). commit.
  3. r110 done -> fold sharding-scaling (@256) into the same story. commit.
  4. BENCHMARKS (the "after" agenda, memory benchmark-targets.md): passkey FIRST — most tractable, = our recall task
     in the standard needle-in-long-context format; build a passkey harness (mech/persist or a new eval) on the
     existing d512/r92 ckpt. THEN BABILong (needs multi-hop CoT-over-memory), THEN FANToM (ToM/dialogue, hardest).
  5. #1 KV/RoPE differentiation: FLAGGED — do NOT blind-implement. It is POSITION-addressed (query needs a position);
     our store is CONTENT-addressed, so RoPE-on-write-order doesn't cleanly help distinct-fact capacity (query can't
     know the slot's write-order). It belongs with passkey/BABILong (position-relative retrieval). Needs the user's
     design call on return. Meanwhile SHARDING (r109/r110) is the working per-store capacity lever.
Between wakeups boxes may idle <=30min (cron cadence) after a chain ends; acceptable. NO Co-Authored-By trailer.
## ===== CAPACITY ASSAULT COMPLETE @09:45 (r103-106) — see below =====
128-fact frontier is GENUINE. vs r104 cold per-layer 45.3%: heads(r105) 43.8% NULL, curriculum(r106, warm from 32f)
45.3% NULL (exact cold ceiling), width d512(r106, 3.4x params) 47.5% marginal (+2, replicated both boxes). L12 depth
dropped (intractably slow). => single shared store saturates ~45% at 128 keys; no cheap lever (heads/curriculum/width)
closes it toward ~95%. WRITTEN UP: r105+r106 RESULTS, README rows, paper limitation #2 finalized + chronicle + appendix,
count 104->106, builds 26pp. Committed 63cbf08. Per-layer capacity investigation (r103-106) DONE.
r107 (map the frontier 64->128): box1 f96, box2 f112, per-layer cold batch16. Known curve: 64=~96%, 128=45.3%; fills
96 & 112 (cliff ~100 or slope?). box2 f112 healthy (step3500, bootstrapping 7.5%) -> ~12:20. box1 f96 DIED at step500
@~09:59 (killed hard, no traceback/EXIT = transient OOM/kill on shared box; box2's heavier f112 fine, GPU now free)
-> RELAUNCHED @10:41 (pid 6573, training, CLEARED step500 = death was transient) -> f96 ETA ~13:15.
@12:10: f112 DONE = **52.2%**. f96 climbing 43%@6000. FRONTIER so far: 64=~96%, 96=~climbing(→~55-60%?), 112=52.2%,
128=45.3% = MONOTONIC SLOPE (steepest 64→96), NOT a sharp cliff. box2 idle after f112 -> launched f80 (fill 64→96
steep segment) @12:10, final ~14:45. HARVEST all r107 points together -> r107 RESULTS frontier curve + paper.
@13:14: FRONTIER filling in (per-layer, cold): 64=~96%, f96 DONE **62.8%**, f112 DONE **52.2%**, 128=45.3%. f80 climbing
(32%@4.5k ->~75%?, ~14:45), f88 LAUNCHED box1 (fills 80-96, ~15:00). CURVE = smooth monotonic slope (NO cliff) =
consistent with "addressing harder with more distinct keys" (user's KV/RoPE framing). PAPER already updated: limitation
#2 + conclusion now frame 128-frontier + 1M as COMPUTE limits w/ KV/RoPE positional-differentiation direction (r67
recency precedent, unlike r89), commit 7d44207 (27pp). HARVEST r107 (f80,f88,f96,f112) when f80/f88 land ~15:00:
write frontier curve into r107 RESULTS + a line/figure in paper. Then arc at natural endpoint (compute-bound).
box1 = user's normally but they said "use both boxes hit harder o7" -> using both. HARVEST r107 -> frontier-shape row
in r106 RESULTS + paper. AFTER: capacity fully mapped; arc at natural endpoint -> user's direction / structural lever
(several stores, needs code) / int4 kernels.

## ===== r104 DONE @01:10 — per-layer write PUSHES the 128-fact wall (0.3% -> 45.3%), not eliminates =====
FINAL: 128 facts mean-write **0.3%** (flatlined) vs per-layer spl_token **45.3%** (150x, deep gates L3-5 @0.51-0.61);
64-fact mean-write baseline 96.3%. So the r88 capacity frontier belongs to the SINGLE MEAN-WRITE value, not the arch:
per-layer LEARNS 128 where mean-write can't, but 45% is PARTIAL (vs ~95% at 32/64) & plateaus -> frontier PUSHED not
removed. Confirms user's prediction (limitation #2). WRITTEN UP: r104/RESULTS.md, README row, paper limitations #2
finalized (0.3->45.3, pushed not eliminated) + chronicle + appendix r104 rows. Paper builds 26pp. Per-layer arc
(r103-104) COMPLETE. box1 = user's (freed). r105 LAUNCHED on box2 @01:15: per-layer f128 with --n-heads 4 (vs r104's 1) = tests the "more
heads" lever from limitation #2's open direction (does it beat r104's 45.3% at 128 facts?). NB same 7.78M params
(heads split d_model), so it tests attention granularity not param-capacity. Cold, 10k steps, ~2.5-3h -> ~04:00.
If >60% heads help; if ~45% heads aren't the bottleneck (points to width/several-stores). Matched baseline = r104 45.3%.
USER @01:15 "use both boxes, hit harder o7" -> box1 FREED again. r106 LAUNCHED box1 (the depth+width capacity levers):
r106_spl_f128_L12 (12 layers = 12 slots/token, 13.3M params) then r106_spl_f128_d512 (wider d512). Both per-layer
f128, cold, batch 4 accum 4 (effective 16; L12/d512 OOM at batch 8 -> use accum). expandable_segments (box1 only).
So the 128-fact assault = 3 levers in parallel: HEADS (r105 box2), DEPTH+WIDTH (r106 box1), all vs r104's 45.3%.
r106 arms ~3-4h each (batch4accum4 slow) -> box1 busy ~overnight; box2 r105 ~04:00. HARVEST each vs 45.3%: which lever
(if any) closes the 128 gap toward ~95%? Feeds limitation #2's open direction. NB batch differs from r104 (accum-matched
effective 16) — note as caveat.
PIVOT @02:15: r106 L12 DROPPED — 12 per-layer reads x accum4 ran ~10h with NO eval in 31min (GPU 99% but intractably
slow). Replaced with FASTER levers: r106_spl_f128_curr32 = CURRICULUM (warm 128 from r103_spl_token.pt 32-fact per-layer
ckpt; the lever that broke every prior fact-count cliff r25/r49/r66; L6 batch16 = fast ~2.5h) THEN r106_spl_f128_d512 =
WIDTH (d512, batch4 accum4). curr32 confirmed warm-start loaded + training. EARLY: r105 heads TRACKS r104 exactly (2.8%
@3500 = r104's 2.8%@3500) -> heads likely WON'T move the gap. Curriculum is now the best hope to close 128. Harvest all
vs 45.3%.
PROGRESS @02:40: r105 heads step6000 26.9% = TRACKING r104 (heads won't move gap, ->~45%). r106 curr32 (curriculum)
started 32.2%@500 (warm-start head start) then DIPPED to 8.4%@1500 = overwriting the 32f solution, re-bootstrapping on
128 (gates wide 0.4-0.6). Final ~04:45 decides: re-climb PAST 45% (curriculum helps) or re-trace to ~45% (warm-start
wasted, 128 a genuine limit even with curriculum). d512 (width) queued after. All harvest vs 45.3%.
PROGRESS @03:10: r105 HEADS step8500 = **42.5%** ≈ r104's 45.3% => HEADS NULL (doesn't move the 128 gap), nearly done
~03:30. r106 curr32 CURRICULUM re-climbing post-dip: 20%@4000 (FASTER than cold r104's ~5%@4000, warm-start helps the
climb), gates very wide (0.71-0.75); ceiling TBD ~04:45 — does it plateau ABOVE 45% (curriculum closes/helps) or ~45%
(warm-start only speeds, doesn't raise). Width d512 queued. So far: heads null; curriculum the live lever.
@03:40: r105 HEADS **DONE = 43.8%** (best 44.1%) ≈ r104 45.3% => HEADS NULL (confirmed, more heads doesn't move 128).
r106 curr32 curriculum step6000 35.9% climbing (tracking r104 now; ceiling ~04:45). PARALLELIZED d512 (WIDTH) onto
box2 @03:40 (idle after r105): d512/L6 per-layer 128f, 26.84M params (3.4x d256), batch4 accum4, NO expandable_seg
(box2) — fit OK, training -> width answer ~06:00. NB box1's run.sh will ALSO hit its queued d512 after curr32 (~04:45)
= redundant with box2's; KILL box1's d512 when curr32 finishes (box2 has it). CAPACITY ASSAULT WRITEUP: bundle r105
(heads null) + r106 (curriculum + width) into ONE combined result when curr32 + d512 land; verdict feeds limitation #2.
Running tally vs r104 45.3%: heads 43.8% (null), curriculum ~TBD, width ~TBD.

## ===== PAPER FULLY WRITTEN UP @~23:00 (104-round saga) — commits 2b29c16 + ec1447c =====
User: "write up paper, weve 104 exps, hyps and results, write all the saga in paper." DONE:
- NEW Appendix (sections/appendix_log.tex): longtable logging ALL 104 rounds (r | hypothesis | result),
  grouped by arc. Added \usepackage{longtable}, \appendix + \input in main.tex.
- Chronicle: new r101-104 subsection (unbounded reach 1M push + per-layer r/w WIN + the detach-bug reverify saga).
  Intro count "~85"->"104" + appendix ref; milestone table result col now wraps (p{}) + rows r101/r102/r103 added.
- Abstract + intro: folded in unbounded store (grow+top-1 ~128k tok) and per-layer r/w +16pt win; count -> 104.
- Builds clean: 26pp, pdflatex x2, no errors, 2 negligible <40pt overfulls.
BOX STATUS: box1 FREED (user needed it, killed r104 f128). box2 RUNNING r104 f128 capacity (batch16, cold @step500,
mem6096) -> r104_spl_token_f128.json; slow (~3h). r103 CONFIRMED both seeds (0.955 exact / 98.3%). r104 = last exp.
RESUME: when box2 f128 lands -> r104 RESULTS (does per-layer write break the 128-fact wall? baseline=0.3%) + README + fold.
r104 SIGNAL @00:10: per-layer f128 BOOTSTRAPPING = BREAKING THE WALL. recall climbs 2.8->5.0->11.9->15.6->23.8% over
steps 3.5k-5.5k (still rising), gates wide (L3-5 @0.47-0.55), vs mean-write baseline FLATLINED 0.3% to step10k. So
per-layer write LEARNS 128 facts where the single mean-write store cannot (r88 wall). UPDATE @00:40: climbed to 42.8%
@step8000, decelerating (35->39->41->43 over 6.5-8k) -> likely ~45-48% final. So wall MOVED SUBSTANTIALLY (0.3%->~46%)
but NOT eliminated (128 still partial vs ~95% at 32/64). Limitation #2 wording = "per-layer PUSHES the frontier, not
eliminates it." Final ~01:06. THEN: write
r104 RESULTS (best+final), README row, and FINALIZE limitations #2 in paper (was "real frontier" -> "per-layer moves
it") with the actual number; commit. This confirms the user's prediction from the limitations discussion.

## ===== LIVE @16:15 — r102 DONE+WRITTEN (525a85b); BOTH BOXES ON r103 =====
r102 DONE: top-1 FLAT 100% to 999 win / 126,873 slots / ~128k tok on BOTH models; dense collapses 87.5/37.5->0%;
rare-token keys weak (50-62% d512 / 12-25% d256 = under-trained embeddings); 1M reach eval-memory-bound (OOMs 16G at
>=380k slots 3 ways), arch is O(1)/query. RESULTS.md + README row + paper longcontext fold + build OK (19pp). DONE.
r103 RUNNING both boxes (per-layer r/w vs read-all/write-mean, cold bpe_recall 10k): box1 baseline_mean (from 16:11)
then spl_token; box2 baseline_mean DONE then spl_residual. ETA ~17:40 (box1 spl_token is long pole; box1 slower ~44min/arm).
PROGRESS @16:40: box1 baseline step6500 recall 26% (bootstrapping, will jump ~8k like box2 did). box2 SPL_RESIDUAL
step6000 recall **0.0%, gates ~0** (baseline at same step was 25%/gates0.28) => residual key NOT bootstrapping, read
gates never open (residual key finds no matches) = strong NEGATIVE signal for residual key (as predicted; abandons the
exact-token match). Not final (6k/10k). The decisive arm is box1 SPL_TOKEN (per-layer write + TOKEN key, not yet run):
if it also just ties baseline, per-layer WRITE doesn't help single-fact recall -> write up negative, move per-layer test
to MULTI-HOP (its real promise: L layers = L hops). Next tick: harvest r103 verdict.
--- (superseded) earlier r102 saga below ---
## ===== LIVE @15:15 — box1: r102 1M frugal retry | box2: r103 RUNNING =====
r102 grow path OOMs at ns3000+: amortized-doubling GB buffers fragments the caching allocator to ~14G, OOMs the next
2G doubling, CPU-crawls. Killed both (2x). box1 NOW on a MEMORY-FRUGAL retry (experiment/r102/push1m.sh): FIXED
pre-sized store (--mem-list ns*130+10000, NO --store-grow) = one alloc, no doubling; ns3000 then ns7800 (~1M tok),
top-1 only. Early: GPU 10.9G (vs 14.2 grow) — helping. If ns7800 fits -> 1M answered; if OOM -> harvest at ns1000
(headline already solid: top-1 FLAT 100% to 999 win / 128k tok BOTH models).
box2 PIVOTED to r103 (per-layer r/w, the user's new default): mech synced to ~/fw, smoked (mem 3048/512 = per-layer
writes across 6 layers OK), launched r103/run.sh box2 = baseline_mean vs spl_residual, cold bpe_recall 10k. box1 will
run its r103 arms (baseline_mean vs spl_token) after the 1M retry frees it.
UPDATE @16:11: 1M PUSH CAPPED. ns3000 (380k slots) OOMs a 16G GPU THREE ways: grow=fragments to 14G; fixed-M=O(nseg*M)
crawl 27min; grow+expandable_segments=maps to free 71MB then fails. The offline eval materialises the whole store to
build+scan it -> can't hold >=380k slots on 16G. This is an EVAL-IMPLEMENTATION limit (arch is O(1)/query, r101), not
capacity. r102 demonstrated reach = 999 windows / ~128k tokens, top-1 FLAT 100% BOTH models (headline solid). To ever
hit 1M would need a streaming/chunked store-scan eval (future). Writing r102 up at the 128k cap.
BOTH BOXES NOW ON r103 (per-layer r/w). box2: baseline_mean DONE (36min, bootstrapped 26->71->~final), spl_residual
RUNNING (step 2000 @16:09, cold, gates barely open). box1: relaunched to r103 (mech synced to ~/research/framework,
parses OK) = baseline_mean (started 16:11) then spl_token. ETA both ~17:20-25. Next tick: harvest r103 (per-layer write
vs mean-write verdict) + write r102 RESULTS at 128k cap + README + paper longcontext.
RESUME TODO next tick: (1) box1 1M retry -> if done, fold ns3000/ns7800 into r102 RESULTS + longcontext; if OOM at
ns7800, write reach as eval-mem-limited & cap. (2) box2 r103 -> when arms done, diagnose (does per-layer write beat
mean-write on recall?), RESULTS + README + paper. (3) launch box1 r103 arms once free. mech r103 code committed b2d4ef4.

## ===== RESUME POINT (2026-09-21 ~13:05, pre-compact) =====
RESEARCH ARC ESSENTIALLY COMPLETE. Paper = 19pp, builds clean, committed (latest paper commit 1ca12cc). Full arc:
store/longcontext/CSA2 (r1-66), managed memory (r76-79), transplant/stateless-weights-stateful-store (r86), multi-hop
CoT 2-5 hops x 16-64 facts TF+autoregressive+controls (r84/85/93/94/95/96), capacity frontier real (r88), gate x depth
curriculum fix (r90), fine-key DEAD (r89/91/92), fuzzy recall geometric-not-semantic (r97/98), SEMANTICS via CoT
parametric hop (r99: 90.8%, broken 2%), GROWABLE store removes r51 cliff (r100: 199 windows). All committed with RESULTS
+ README rows + paper folds. NO Co-Authored-By trailer (CLAUDE.md).
r101 DONE + WRITTEN UP (@13:15, committed 2676a4c): grow + top-1 sparse = unbounded store at FLAT **100%** to 199
windows (25,273 slots, 1 slot), dense decays 85->60%. r99 s1 (90.6%) + L32 (85.8%) folded (hop1 100% throughout).
LIVE NOW (@13:30): r102 RUNNING on BOTH boxes (1M-token push): box1 d512_L2 (via chain, HEALTHY: store grows to
25273/126873 slots, top-1 recall meaningful), box2 r92_recall_exact (d256L6). Each sweeps nseg 200/1000/3000/7800
(~1M tokens) x {--rare-keys, normal}, grow + top-1. rare-key arm = does the STORE reach 1M; normal arm = collision
ceiling (r52/r89). The 7800-window runs are slow. box1 skips-if-done (r102 SKIP logic).
**BOX2 WAS BROKEN & FIXED @13:29:** box2's mech/config.py + hippo_layer.py were STALE (pre-r100, no store_grow/_grow;
only persist.py had _set_grow), so its store never grew (stuck 512, ring lapped) -> ALL cells 0.0%. Fixed: rsync'd
mech/*.py to box2:~/fw/mech (parses OK, grep confirms grow code), wiped the invalid all-zero results/logs, relaunched
detached. Confirmed store now grows to 25273 slots. Watch: box2 recall is weaker (rare_ns200 top-1 25%); its model is
cold-recall d256L6. EARLY SIGNAL both boxes: rare-key arm UNDERperforms normal (box1 rare_ns200 50% vs normal 100%) —
counter to the "rare=store reaches 1M cleanly" prediction; likely rarest-word embeddings are under-trained -> weak keys.
Noisy (batch2 iters4 = 8 examples, 12.5% granularity); confirm at full sweep.
OOM FIX @14:45: the dense graded(all) read materialises ~[M x d] intermediates and OOM'd at ns3000 (380k slots) on
the SHARED GPU (box1 had 976MB free under another user's load), then CPU-crawled 17min with no output. Both boxes
were stuck the same way. FIX: killed both chains by PID (box1 2583111/2584026/2584029; box2 580885/582185/582188),
switched r102/run.sh big cells (ns>=3000) to TOP-1 ONLY (TK=1; top-1 is the headline, dense collapse already shown at
ns1000=0%), relaunched both. ns200/ns1000 JSONs preserved (skip-logic). box1 GPU now FREE (15.4G) -> should finish
ns3000+ns7800 fast; box2 only 3.36G free -> may OOM the ns7800 store (988k slots ~4G d256) and cap at ns3000 (honest:
reach is GPU-limited there, not architecture-limited). r102 headline stands regardless: top-1 FLAT 100% to 999 win.
PROGRESS @14:00: box1 5/8 (on rare_ns3000), box2 3/8 (on rare_ns1000), both HEALTHY. KEY FINDING firming up:
NORMAL-key top-1 = FLAT 100% out to 999 windows / 126,873 slots (~128k tokens) on BOTH models (box1 d512L2 + box2
d256L6) -> extends r101's flat-100% (199 win) 5x further. DENSE read collapses with scale (normal_ns1000 graded=0.0%
vs top-1 100%) = the sparse-wins story at 1000-window scale. rare-key arm stays BELOW normal (box1 62.5% / box2 low)
= rarest-word embeddings make weak keys, a key-quality effect not a capacity limit. ns3000/ns7800 (toward 1M) pending.
**NEW CONVENTION (user @13:29, r103+):** switch to PER-LAYER read/write against ONE shared store, replacing
read-early/write-late ("last write first read"). Every layer r+w the shared store; store stays shared (not per-layer).
r102 stays on OLD scheme (box1/box2 comparability); implement per-layer r/w in model.py+hippo_layer.py behind a config
flag for r103. See memory per-layer-rw-convention.md.
r103 STAGED + SMOKED @14:05 (code committed): shared_per_layer implemented (mech/config,model,hippo,train) behind a
flag, default off = byte-identical old path. CPU smoke OK: per-layer write lands 248 slots vs mean-write baseline 62
(~L x), grow absorbs it, baseline unchanged. r103 = A/B cold bpe_recall (10k steps, 32 facts, r92 hp):
box1 baseline_mean vs spl_token; box2 baseline_mean vs spl_residual. run.sh + HYPOTHESIS written.
*** LAUNCH PREREQ: boxes DON'T have r103 mech yet -> when r102 frees a box, FIRST sync mech/*.py to it
   (box1: git pull or rsync; box2: rsync via b2ssh to ~/fw/mech), smoke --steps 5, THEN launch r103/run.sh. ***
RESUME TODO: (1) when r102 lands -> r102/RESULTS (1M push: rare-key reach vs collision ceiling; note rare<normal
surprise) + paper longcontext. (2) THEN launch r103 (sync mech first!) — first per-layer-r/w round, new default.
Cron 34e14d22 (:07/:37) drives the loop. box1=ssh box1 (~/research/framework); box2=b2ssh wrapper (~/fw).

## (older) Now (2026-09-20 ~12:45) — r84 DONE+committed (CoT-over-memory WORKS). r85 SCALE-UP running, both boxes healthy.
Progress @12:45 — 2 jobs DONE, both boxes on job 2:
- **box1 wd_d256_L6 DONE = 0.3%** (FINAL, never transitioned: 0.0->0.3 over 12k; READ GATE stayed shut a0=0.035
  despite 228 slots written). FINDING: the managed store (--write-op-gate) was only ever validated WARM (r76-79);
  this is its first COLD test and d256L6 did NOT cold-bootstrap it. Reframes the box1 study as "which shape can
  cold-bootstrap the managed store?" — d256L6 no; DEPTH (d192L12) may. Matched study, let it run. Now: wd_d384_L3.
- **box2 store_f64_m2048 DONE = 95.9%** (final, 64 facts, warm r79) — bounded gated store holds at 64 facts/M2048.
  Now: store_f128_m4096 (step 2000, recall 36% climbing).
qrun alive on both. NEXT: as each sub-study (box1 arch set / box2 scale set) drains, pull its result JSONs (targeted,
NOT all of experiment/) and write experiment/r85/RESULTS.md. If ALL box1 shapes fail cold-bootstrap -> that is the
finding + motivates an r86 gate-free/warm arch study. If box1 depth transitions -> width-depth answer stands.

DONE so far (finals): box1 wd_d256_L6 **0.3%** (deep/square, no cold-bootstrap), wd_d384_L3 **95.3%** (wide/shallow,
bootstraps). box2 store_f64_m2048 **95.9%** (64 facts ok), store_f128_m4096 **26.1%** (128 facts = capacity frontier).
r85 DONE + WRITTEN UP (@18:11): all 10 jobs drained (box1 16:54, box2 15:51). RESULTS.md + README rows (r85/r86/r87)
+ chronicle scale-up subsection folded to finals + milestones row. Paper now 18pp, clean. FINDINGS:
- (1) MULTI-HOP SCALES to 32 facts: 2-hop **96.4%**, 3-hop **95.2%** (r84 was 16 facts), each hop ~98%.
- (2) CAPACITY FRONTIER: 64 facts **95.9%** ok, 128 facts **26.1%** (peaks ~46% then destabilises) = 64-128 wall.
- (3) WIDTH×DEPTH monotone in DEPTH: shallow bootstraps the managed store (d512_L2 **96.7%**, d384_L3 **95.3%**), deep
  FAILS (d256_L6 **0.3%**, d192_L12 **0.2%**, read gate shut). First COLD test of the write-op gate (r76-79 were warm)
  -> the gate can't bootstrap in a deep stack. (bootstrap result, not exploitation -> r87 tests gate-free.)
- (4) LM BENCH: memory on **ppl 3.81** (val 1.338) vs off **3.85** (1.347), gain in later segments as store fills (=r34).

r86 DONE + WRITTEN UP (@19:50): TRANSPLANT — thesis CONFIRMED both halves. Same frozen model + 2 stores: own
100/96.9%, wrong-store swap **0.0%**, scrambled 3.1% -> store IS the state. FOREIGN: modelA own 98.8% / modelB own
100%, each reads the OTHER's store **0.0%** both ways (facts present, unreadable) -> store bound to its weights (keys =
writer's embeddings). RESULTS.md + README r86 row + paper transplant.tex (numbers folded, 18pp) + committed.
r87 RUNNING (box1, launched 18:13): gate-free width×depth — arch2_d256_L6 DONE (~85% vs gated 0.3% -> GATE was the
blocker), now d384_L3; then d192_L12, d512_L2 (~23:00).
r88 RUNNING (box2, launched 19:43): capacity curriculum f64->f128 16k (~22:00) — is the 64-128 wall under-training?

r89 STAGED + AUTO-LAUNCH ON NEXT FREE BOX (user: "queue it so it auto runs"). DG pattern separation via a FINE
(bigram) key. Code committed 64124e9 (config fine_key/fine_key_alpha; model.py bigram q=emb(t)+alpha*emb(t-1);
tasks.py bpe_collision_batch; train.py wired --task bpe_collision --collisions-per-key --scramble-context --fine-key).
SMOKE PASSED on box2 (both exact+fine at steps 2, exit 0). mech/*.py + experiment/r89/queue.txt + r79 ckpt present on
BOTH boxes. Queue = 7 jobs (C4 exact/fine/scramble, C2 exact/fine, C8 exact/fine), warm r79, 6k each (~4h).
LAUNCH when a box frees (box2 ~22:00 first): `cd <ROOT> && setsid bash experiment/r85/qrun.sh <ROOT> experiment/r89`
(ROOT=~/fw for box2, ~/research/framework for box1). Predict: exact ~1/C, fine >> exact (high), scramble ~chance.
Then r89/RESULTS + README + paper (fold into limitations/chronicle: exact-match now separates collisions via context).

r87 DONE + WRITTEN UP (@22:40, cron 34e14d22 live :07/:37): gate-free ALL 4 shapes bootstrap (d256_L6 84.7, d384_L3
97.5, d192_L12 97.0, d512_L2 98.8) vs r85 gated (0.3/95.3/0.2/96.7). => r85's "depth kills cold-bootstrap" is a
GATE x DEPTH interaction; the plain store is depth-robust. RESULTS + README + paper (chronicle+limitations, 18pp) +
committed. box1 now IDLE -> launching r90.
r90 (box1): CURRICULUM for the gate — warm the GATE from the r87 plain ckpts (arch2_d256_L6.pt, arch2_d192_L12.pt, the
shapes that failed cold gated) + --write-op-gate: does the gate learn when the store is already bootstrapped? If recall
stays high + mem drops to O(facts) -> curriculum fixes the gate x depth cold-start.
r88 DONE + WRITTEN UP (@23:10): 128-fact wall REAL — warm-from-f64 peaks 47.2% then COLLAPSES to 3.1% (curriculum
did NOT help, destabilised). RESULTS + README + paper limitations (18pp) + committed.
r89 RUNNING (box2, auto-launched 22:45 via chain OK): col_C4_exact DONE = **22.0%** (= 1/4 collision draw, baseline
confirmed); col_C4_fine RUNNING (the fine-key test); then C4_scram, C2/C8 exact/fine (~3.5h).
r90 RUNNING (box1): r90_gate_d256_L6 step 3500 recall **74.1% mem 228/1024** -> the GATE learns WARM (vs 0.3% cold)
AND bounds to O(facts) -> curriculum fixing the gate×depth cold-start looks promising. Then r90_gate_d192_L12.
NEXT: r89 done -> RESULTS + paper (DG separation: does fine-key lift 22% -> high?); r90 done -> RESULTS (curriculum).

UPDATE @23:40:
- r90 gate d256_L6 DONE = **94.8% mem 228/1024** (warm from r87 plain 84.7 + gate). => the GATE LEARNS WARM (vs 0.3%
  cold) AND bounds to O(facts). CURRICULUM FIXES the gate×depth cold-start. r90_gate_d192_L12 (deep) running.
- r89 col_C4: exact **22.0%** (=1/4 draw, baseline confirmed) BUT fine **0.2%** (COLLAPSED, worse than baseline).
  Not "no lift" — total breakage => likely a WARM-START CONFOUND: bolting a bigram key onto r79's single-token-key
  read path breaks matching (exact arm keeps single-token keys, works). NOT a clean DG test. Let queue finish
  (scram/C2/C8 — if all fine arms ~0, systematic). Write up as NEGATIVE (do NOT re-tune); clean test = COLD fine-key
  from scratch (r91, mirrors how r87 corrected r85's warm-only gate). Both boxes busy; healthy.

r89 + r90 DONE + WRITTEN UP + COMMITTED (@02:45 Sep 21):
- r90 CURRICULUM FIXES gate×depth: warm the gate from r87 plain ckpts -> d256_L6 **94.8%**, d192_L12 **95.6%** (cold
  gated 0.3/0.2), mem ~228 (O(facts)). The gate×depth cold-fail is a curriculum artifact. RESULTS+README+paper.
- r89 collision: exact tracks 1/C (C2 43.6 / C4 22.0 / C8 1.2) = real collision limit confirmed; fine-key COLLAPSED
  to 0.2% at ALL C (warm-start confound — bigram key on r79's single-token read path breaks matching; NOT a fair test).
  RESULTS+README+paper.
r91 RUNNING (both boxes, launched 02:44 via qrun): COLD fine-key study — exact vs fine trained FROM SCRATCH (no warm
confound). box1 queue.txt = C4 exact/fine + C2 exact/fine; box2 = C8 exact/fine + C4 scram. 12k cold each (~4h box1,
~3.5h box2). If cold-fine >> cold-exact -> DG separation works; if cold-fine also collapses -> bigram incompatible
with the frozen-identity exact match. NEXT: r91 done -> RESULTS + paper DG verdict. Open threads: autoregressive
multi-hop eval, fuzzy/LSH recall, int4 kernels.

UPDATE @03:45: r91 KILLED + WRITTEN UP — cold collision does NOT bootstrap (exact_cold C4/C8 both **0.2%** at 12k,
vs warm r89 22%/1.2%): can't test the fine-key cold. Methodological negative (r91/RESULTS). PIVOT -> r92.
r92 RUNNING (both boxes, launched 03:44): fine-key done RIGHT — bootstrap the bigram read path on EASY recall WITH
--fine-key from step 0, then warm the collision from it (a COMPATIBLE warm start, fixing both r89's confound and r91's
no-bootstrap).
  box1 (fine): r92_recall_fine_cold [CRUX: does the bigram bootstrap on recall?] -> r92_col_C4_fine_warm.
  box2 (exact, matched): r92_recall_exact_cold -> r92_col_C4_exact_warm (~22% = 1/C baseline).
  ~90min/box. PASS = fine collision >> exact 22%. If recall_fine_cold FAILS (~0) -> the bigram key can't bootstrap ->
  DG-via-bigram is dead; write up and move to other threads (fuzzy/LSH recall, autoregressive multi-hop eval).

r92 DONE — FINE-KEY DEAD (@04:45): recall exact cold **84.7%** vs recall fine cold **0.0%** (gate shut), same settings
-> the bigram key can't bootstrap the exact-match read; ANY key enrichment moves off the raw token embedding and breaks
exact match (= contextual r42 / projected r31). Fine-key / DG-via-bigram CLOSED. RESULTS + README + paper limitations.
The collision limit (1/C, r89) stands unsolved by key-enrichment; the fix must relax the COARSE term at READ (make the
LSH/ann pool carry recall), NOT enrich the key.
r93 RUNNING (both boxes, launched 04:45): multi-hop FRONTIER — box1 4-hop@32, box2 2-hop@64, warm r79, +broken
controls (~1h). How deep / how many facts does CoT-over-memory reach?
NEXT: r93 done -> RESULTS + paper (multi-hop depth/fact-count limits). Open threads: fuzzy/LSH recall (relax coarse
term at read — the r92 lesson), autoregressive multi-hop eval (the honest number), int4/hash kernels.

r93 DONE + WRITTEN UP (@06:10): multi-hop FRONTIER EXTENDS — 4-hop@32 **91.6%** (per-hop 98/98/98/98, broken 0.0% =
97/97/95/**0**), 2-hop@64 **96.9%** (99/98, broken 0.0% = 88/**0**). CoT-over-memory now mapped 2/3/4 hops × 16/32/64
facts, EVERY point broken-controlled at chance. RESULTS + README + paper chronicle. Committed.
r94 RUNNING (both boxes, launched 06:10): one more frontier step — box1 5-hop@32, box2 3-hop@64, +broken controls (~1.3h).
AR EVAL BUILT (@06:40): persist.py `chain_ar` — autoregressive multi-hop (feed the model its OWN emitted intermediate
per hop). Committed fbe9001, synced both boxes; restore() now loads bpe_chain ckpts. NOT yet run (smoke first).
r94 DONE + WRITTEN UP (@07:45): 5-hop@32 **89.7%** (per-hop 97/98/98/98/98, broken 0.0%=.../0), 3-hop@64 **95.2%**
(broken 0.0%). Multi-hop CoT-over-memory now mapped **2-5 hops × 16/32/64 facts**, EVERY point broken-controlled.
RESULTS+README+paper chronicle. Committed. chain_ar SMOKE OK (4-hop AR final ~97% on 32 ex — eval functional).
r95 DONE + WRITTEN UP (@08:10): HONEST autoregressive numbers MATCH teacher-forced to the digit — 2-hop@64 **96.9%**,
3-hop@64 **95.2%**, 4-hop **91.6%**, 5-hop **89.7%** (n=640; per-hop ~0.98 geometric). CoT-over-memory confirmed
END-TO-END feeding the model's OWN intermediate. Closes the multi-hop caveat. RESULTS + README + paper (memory.tex +
chronicle) + committed.

RESEARCH STATUS: core arc COMPLETE — store/longcontext/CSA2 (r1-66), managed memory (r76-79), transplant (r86),
multi-hop CoT 2-5 hops × 16-64 facts TF+AR+controls (r84/85/93/94/95), capacity frontier real (r88), gate curriculum
(r90), fine-key DEAD (r89/91/92). Paper 18pp comprehensive.
NEXT FRONTIER (needs a design call — may hit the exact-match wall the fine-key did): FUZZY/SEMANTIC recall — relax the
COARSE key at READ (make the LSH/ann pool carry recall for near-keys) so a paraphrase retrieves. This is the one gap
between dictionary and hippocampus. Worth a speak-through with the user before building (r92 showed key-enrichment fails;
read-side relaxation is untested but non-obvious). Also open: int4/hash kernels (engineering, make measured wins literal).

UPDATE @08:40: r96 seed-1 replication matches seed-0 EXACTLY — 4-hop@32 s1 **91.6%**, 3-hop@64 s1 **95.2%**.

FUZZY-RECALL PROBE (user idea "i doubt fuzzy is alr there" — probe existing models, no training): added
persist.py `fuzzy` mode — fill store with exact-key facts, query with the k-th NEAREST EMBEDDING NEIGHBOUR of each
key (a token never shown, fact tokens excluded), measure recall. On r87 arch2_d256_L6 (85% exact): rank0(exact) 85.0,
**rank1(cos0.94) 36.2%**, rank2 25.0, rank5 15.0, rank10 11.2 (chance ~3% at 32 facts). => FUZZY IS PARTIALLY EMERGENT:
the "exact" match is really argmax-cosine, so a near-embedding query lands on the right slot ~12x chance — but weak
(36 vs 85) and it's SURFACE-embedding-near, not proven semantic. n=80 (small). Committed as r97 code.
NEXT (when r96 frees boxes ~09:25): FIRM the fuzzy probe (more iters + 2nd model + settle-sharpness sweep to test the
exact-vs-fuzzy trade), write up r96 (replication) + r97 (fuzzy emergent). Awaiting user direction: document / widen /
test-semantic.

UPDATE @10:50 — FUZZY-RECALL THREAD RESOLVED (r97/r98) + testing the CoT-semantics reframe (r99):
- r97: fuzzy (near-embedding-neighbour) recall is PARTLY EMERGENT in the plain store (nn-1 33-73%, arch-dependent,
  wide/shallow widest), KILLED by the gated store (0.2%). The exact match is a soft argmax-cosine NN lookup.
- r98: but it's NOT semantic (synonym query = chance ~4.2% across all 6 models; embeddings barely separate synonyms
  from random, cos 0.91 vs 0.87) and can't be widened by softening (softer read COLLAPSES the basin; sharp argmax is
  what makes it work, saturates ~75% at beta>=64). Store-side semantic recall = DEAD END.
- User reframe "multilayer cot should do semantics": semantics belongs in REASONING, not store matching.
- r99 RUNNING (both boxes, launched 10:48): bpe_semchain — a CoT with a PARAMETRIC first hop (fixed syn->canon
  lexicon learned in WEIGHTS, not stored) + store second hop (canon->val). box1 main+broken(control), box2 seed1+L32.
  If hop1 (syn->canon) is learned and hop2 reads -> semantic recall by reasoning though the store can't fuzzy-match.
  Committed: persist fuzzy/semantic/settle-beta probes; bpe_semchain task. Paper limitations folded (18pp).
NEXT: r99 done -> RESULTS (does the parametric CoT hop work?) + paper. This is likely the last open thread.

UPDATE @11:10 — r99 CRUX POSITIVE (mid-run, both boxes step ~3k/10k, stable): char **100/91** -> hop1 (syn->canon,
PARAMETRIC, learned in weights) = **100%**, hop2 (canon->val, STORE) = 91%, joint **90.8%**. Both seeds identical.
=> "multilayer CoT does semantics" CONFIRMED: the model learns the fixed synonym lexicon in its WEIGHTS and applies it
in CoT to bridge a synonym to the exact store, though the store can't fuzzy-match (r98). Await broken control (random
lexicon -> hop1 must fail) + final. Then write up r99 + fold into paper (semantic recall = reasoning, not matching).

r99 DONE + WRITTEN UP (@12:35): MULTILAYER CoT DOES SEMANTICS (controlled). main joint **90.8%** (hop1 syn->canon
parametric **100%**, hop2 canon->val store 91%); broken-lexicon control hop1 **2%** (unlearnable map fails) while hop2
stays 91% -> hop1 is genuinely LEARNED semantics. Semantic recall = reasoning (weights+CoT), NOT store matching (chance,
r98). Resolves the fuzzy/semantic thread. RESULTS + README + paper (limitations) + committed. box2 s1+L32 still running
(~13:30, fold seed/lexicon-size next tick).

PAPER WRITTEN UP (@13:00, user ask): 19pp, r96-r100 folded — growable store in longcontext, fuzzy/semantics
closing-threads subsection in chronicle + milestones, abstract updated (~100 experiments). Committed 1ca12cc. Builds clean.
r100 DONE: growable store removes the r51 cliff — fixed ring 0% by 7 windows, grow holds to **199 windows / 25k tokens
(62.5%, 25,273 slots)**. Residual decay = dense read distractor mass (r62), not growth.
r101 RUNNING (box1): grow + top-1 SPARSE read together, swept to ~200 windows. Smoke: at 63 win, top-1 **100%** (1 slot)
vs dense 77.5% -> grow removes cliff, sparse removes decay = unbounded store at FLAT recall. box2 r99 s1/L32 finishing.
NEXT: collect r101 (flat recall to 199 win) + r99 s1/L32 -> fold into paper. Research arc essentially COMPLETE.

GROWABLE STORE BUILT (r100, user "grow like a KV cache" = append not lap): config store_grow; EpisodicStore._grow
(amortized doubling of all per-slot buffers) + grow-aware _alloc/write_many + live-size valid_mask; train --store-grow;
persist --store-grow (context/sparse) via _set_grow. Syntax OK, NOT yet smoked. box1 idle -> smoke then launch r100:
context-bridge with --store-grow vs fixed ring on a trained recall ckpt (r62/r50 bpe64) -> does grow remove the r51
cliff + hold recall at 200+ windows? Plain write path only (gated bounds itself).

UPDATE @09:40: r96 box1 DONE (4hop_s1 91.6% + broken; = seed0). box2 r96 broken control finishing (~09:44). r97 fuzzy
FIRM-UP launched on box1 (idle): persist fuzzy on arch2_d256_L6 + arch2_d512_L2, nn-ranks 0-10, iters 20 (n=640) —
firms the 36% + tests arch dependence. box2 r97 (fuzzy on r92_recall_exact + store_f64) launches when it frees.
NEXT tick: write up r96 (both boxes) + collect r97 fuzzy numbers.
NEXT: r86 done -> fold foreign numbers into paper transplant.tex + write r86/RESULTS. r87 done -> r87/RESULTS +
gate-vs-gate-free contrast. Autoregressive multi-hop eval still queued (the honest CoT number).

UPDATE @19:15 — box1 r87 arch2_d256_L6 (gate-FREE) step 11500/12k recall **83.9%** (vs gated wd_d256_L6 0.3% FINAL) ->
gate-free d256L6 DOES cold-bootstrap; the write-op GATE was the blocker, CONFIRMED. ~done, then d384/d192/d512.
box2 r86_modelA DONE **98.8%** (strong independent model A); r86_modelB step 3500 recall 10% (cold, climbing per r66);
then swap eval ~20:00 = the transplant answer (foreign A<->B).
- CAPACITY pair COMPLETE: store_f64 **95.9%**, store_f128 **26.1%** (frontier between 64-128 facts warm/8k).
Width-depth law firming: shallow+wide cold-bootstraps the gated store (d384L3 95.3), deep/square does NOT (d256L6 0.3,
d192L12 0.0). WRITE-UP PLAN: one comprehensive r85 RESULTS + README row + fold into chronicle's scale-up subsection +
commit, when the whole queue drains (~18:00). Don't write piecemeal.

PAPER (user asks: "writeup everything as chain of thought/exp/results" + "feels lil small, expand"): now 17pp,
builds clean. Sections added: memory.tex (r67/r76-79/r84), chronicle.tex (FULL ~85-round arc, 9 threads + milestones
table), sleep.tex (r26-33 consolidation/replay/compression), transplant.tex (the thesis), limitations.tex (r85
frontier + open threads). Thesis woven into abstract + intro: "WEIGHTS STATELESS, STORE STATEFUL" (user's framing).

THESIS DEMONSTRATED (r86 same-model half, run now on store_f64): one frozen model, two stores -> answers each store's
query ONLY when that store is loaded. store1/query1=100%, store2/query2=93.8%, SWAP (wrong store)=0.0%, scrambled=0.0%.
=> the store IS the state; swap store = swap memory, no weight changed. In paper transplant.tex Table.
Added mech/persist.py `swap` mode (in-process transplant, bypasses the fingerprint) + --run-b/--ckpt-b.

r86 CREATED (experiment/r86/, HYPOTHESIS+run.sh) — the FOREIGN half: 2 INDEPENDENT d256L6 recall models (seed 0/1,
plain store so they cold-bootstrap per r66), then swap eval incl foreign A<->B transplant (predict: chance = memory
bound to its weights). LAUNCH after r85 drains: `setsid bash experiment/r86/run.sh box1` (box1 free ~18:00). Then fold
foreign numbers into transplant.tex + write r86/RESULTS. Fold r85 finals into chronicle scale-up subsection too.

**r85 = the scale-up (user: "scale up big, many models, bigger stores, width vs depth, chase benches; QUEUE
everything cuz prone to network blip").** Durable resumable job queue: `experiment/r85/qrun.sh <ROOT> experiment/r85`
runs `experiment/r85/queue.txt` (one `name|args` job per line), SKIPS any job whose result JSON exists (so a
blip/sleep just resumes at the unfinished job), single-instance lock, launched detached via setsid. Safe to relaunch
anytime (e.g. on box wake). Smoke-tested all 3 representative jobs at --steps 2 (all exit 0) before launch.
- **box1 queue (width vs depth):** bpe_recall cold, 32 facts, store 1024, 12k steps — `wd_d256_L6` / `wd_d384_L3` /
  `wd_d192_L12` / `wd_d512_L2` (~fixed params L·d²). Q: which model shape exploits the store best?
- **box2 queue (capacity + multihop-at-scale + LM bench):** `store_f64_m2048`, `store_f128_m4096` (bpe_recall warm
  r79 — does the bounded gated store hold at 64/128 facts + bigger M?); `chain_f32_2hop`, `chain_f32_3hop` (bpe_chain
  warm r79 — does CoT hold at 32 facts?); `lm_hc_on` vs `lm_hc_off` (task lm, PG-19 char, honest ppl delta = does the
  memory block help REAL LM? the r71 benchmark). All warm from r79 except LM (cold) and the arch study (cold).
NEXT: paper CoT-over-memory section (track 2, in progress while queues run); autoregressive multi-hop eval; then
pull r85, write RESULTS, chase more benches. Relaunch qrun on any box that sleeps (resumable).

## (older) Now (2026-09-18 ~14:15) — OBJECTIVE CLOSED on real language (r51 done). r50 16f rung finishing (~10 min).

**r51 DONE — the objective is closed on real language.** Offline probes on r49/r50 checkpoints (box2):
- **Persistence:** reloaded in a FRESH process **100%** == in-process 100%; mismatched store **0.0%**. State is a
  property of the file, not the interpreter.
- **Exceeds context window:** empty 128-tok prompt, store bridges ≈**M/128 windows** — M=512→3win 98.8%,
  1024→7win 98.8%, 2048→15win 93.8%, **M=4096→31 windows (3968 tok) @90.0%**. Hard cliff exactly at `slots==M`
  (ring buffer laps seg-0 facts). **Span scales LINEARLY with store size M, no retraining (buffers non-persistent).**
- Write-ups DONE: `r50/RESULTS.md`, `r51/RESULTS.md`, README rows r50+r51, OBJECTIVE.md "CLOSED" section + recipe.
- **r50 fully done:** seed-2 ladder 8f/16f **100%** final (matches seed 1), 4f cold 35.3% (the bootstrap base),
  64f **99.1%**, scrambled control **0.16%**. Nothing owed on r50/r51.

**r52 DONE — r51's cliff is the eviction POLICY, not capacity.** Offline prune (`mech.persist evict`): ring cliffs
to 0% (=r51), **oracle** (protect the 127 fact slots, evict oldest prose) holds **~98-100% out to 31 windows at
M=512 AND M=256**. A 256-slot store (2 windows raw) bridges 31 windows. Effective reach = `F/128` + working set,
NOT `M/128`. Bonus: oracle M=256 > M=512 (less prose = less read interference). Recipe gains a 2nd axis: **keep the
right slots**. Written up: r52/RESULTS, README, OBJECTIVE recipe. Committed.

**r53 DONE — surprisal is a real label-free signal but has a ceiling below the oracle.** 64f/M=512: surprisal
tracks oracle within ~1pt out to **15 windows (97.5%)** vs ring 0%; but collapses to 0% at 31 windows (and by 15 at
M=256) — PG-19 prose's own high-surprisal tokens (rare words/names) outrank facts once prose piles up. Fewer facts
is HARDER (16f 75.6%@15win < 64f 97.5%). Oracle flat 98-100% (confirmed across fact count+seed, box1 cross). Written
up: r53/RESULTS, README, verdict PARTIAL. Committed.

**r54 DONE — a margin law for surprisal, plus the repeat-suppression failure mode.** 64f: surprisal meets oracle
above **M*≈512 (15-win) / ≈1024 (31-win)**; M* ≈ facts + prose-tail, grows with bridge. 16f: **FALSIFIED —
plateaus ~75% at any M** (until eviction off), because surprisal scores NOVELTY and tiled fact repeats go
low-surprisal → evicted; 64 distinct facts fill the window once so all survive. Written up: r54/RESULTS, README,
OBJECTIVE recipe. Committed. Both boxes idle.

**r55 DONE — dedup lifts the 16f plateau to the oracle, but only with headroom.** `surprisal_dedup` (keep highest-
surprisal copy per token) breaks the ~75% plateau to **100% at M≥1024 (15win) / ≥1536 (31win)** — repeats no
longer dilute each fact's first occurrence — but ties ~73% at tight M≤768 and is marginally worse there (removing
redundant fact copies makes each more fragile). 64f control: dedup ≈ surprisal (no harm). Written up: r55/RESULTS,
README, OBJECTIVE. Committed.

### The offline eviction thread (r52→r55) is CONCLUDED and fully characterized:
- r52: cliff is policy not capacity (oracle bridges 31 win from 256 slots).
- r53: surprisal is a real label-free signal (tracks oracle to 15 win @ 97.5%, 64f).
- r54: margin law M*≈facts+prose-tail; plain surprisal scores NOVELTY → 16f tiled repeats evicted (plateau ~75%).
- r55: dedup recovers the oracle given headroom, but NOT at tight M — facts still lose the top-M contest to prose.
- **Shared limit of ALL offline (read-side) policies:** they prune a store that ALREADY holds the prose, so at
  tight M facts compete with prose it should never have stored. The oracle only wins by being *told* which are facts.

### r56 DONE — placement is a FLAT TIE on the shared store. 2×2 matrix (read start/all × write last/all):
wlast 99.8, wall 100, rall_wlast 99.2, rall_wall 100; replicates 100; scrambled 3.3% (chance). Reading/writing at
every layer buys nothing over read-early/write-late — the layer-0 token-embedding key does the matching wherever
the value lives. Built `--write-layers`. Completes the write half of the user's msg 30/39 ask (read half = r41).
Written up: r56/RESULTS, README. Committed.

### r59 DONE — the fact-count cliff is the SINGLE layer-0 site; multi-layer read/write escapes it COLD.
128 symbols, MQAR cold: single-layer 8f 99.7 → 16f **9.1** → 32f 4.2 (r24 cliff). shared-symmetric (read+write@all,
ONE store) **99.4 / 96.2** @16/32. per-layer **92.7 / 71.6 / 100** @16/24/32 (escapes but seed-noisy). 3rd deadlock
escape (beside r25 curriculum). Qualifies r56 (placement matters AT the cliff, tie only below it). **Per-layer does
NOT beat shared — shared is cleaner + cheaper (1 store vs 6) → leans pro-CSA2 (share one KV state across layers).**
Written up: r59/RESULTS, README. Committed. Both boxes idle. **User's bet half-confirmed: per-layer escapes, but
doesn't beat shared.** Opens: 2-3 seeds for the per-layer noise; re-check on BPE/language; single-layer objective
placement should probably move to read+write@all given this.

### (was) r59 RUNNING — the REAL breaking-point sweep (128 symbols). r57/r58 used the 32-symbol DEFAULT = easy regime:
everything <=32 facts hit 100% (per-layer AND shared tied), f64 CRASHED (64>32 symbols, IndexError). That could not
locate the cliff (r24's cliff needs 128 symbols so facts << key vocab). FIXED with --n-symbols 128. box2: cmp
(single-layer baseline sl + shared sh @ 16/32). box1: pl (per-layer @ 8/16/24/32), chained after r57. r58 KILLED
(moot in easy regime). **User's bet: per-layer store holds LONGER than the single shared layer-0 key.**

### ⭐ NORTH-STAR RESULT — ACHIEVED (r61 DONE, 2026-09-18). Store ≫ window AND query touches ONE slot.
At M=4096 the store holds **31 windows (3968 tok)**; **top-1 read = 100% touching 1/3937 slots**, BEATING dense
(90%). top-1 = 100% at 7/15/31 windows (dense 98.8/93.8/90.0 — dense decays with store size, top-1 immune). Written
up: r61/RESULTS, OBJECTIVE ⭐ section, README. **r60 DONE** — the r59 escape is seed-robust (sh 16f 100/100, 32f
91.6/100; pl 16f/24f 100/100 — the 71.6% dip was noise). Both boxes IDLE. Everything committed.

### r62 DONE — north-star HARDENED & bulletproof. Shuffle control → chance (0.0/0.6%, both models). Scales to
**63 windows (M=8192, 8001 slots): top-1 = 100%** touching 1 slot; dense decays 93.8→90→82.5. Cross-model bpe16s2
needs top-2 (binding spans 2 slots) → 100%. **O(1) slots/query (1-2), store-size-independent; small-k sparse ≥
dense.** Written up: r62/RESULTS, README. Committed. Both boxes IDLE.

### r63 DONE — multi-layer escape PARTIALLY transfers to real language. BPE cold: 16f 53.4→**86.9%**, 32f
16.1→**50.0%** (read+write@all vs single-layer; +~33 pts). Big lift but NOT MQAR's full escape — BPE has a 2nd
bottleneck (multi-token/embedding competition, r36/r47) only the curriculum clears. Written up: r63/RESULTS,
README. Committed.

### r64 DONE — CSA2 shared store CORE ACHIEVED (v0.5). Candidate-pool read makes O(1)-slot -> O(1)-compute:
**token** index scores ~1.9 of 8001 slots at 63 windows, recall = dense (81.9 vs 82.5); **ann** (LSH) ~M/12, also
= dense; **cross-layer reuse FREE** (pool once at anchor = per-layer rebuild, 100% both). `none`=dense=r62 (no
regression). Built read_pool (none|token|ann) + key_tok buffer + pool probe; fixed a stale-cache bug (reuse is
across LAYERS not segments). Written up: r64/RESULTS, README. Committed. Both boxes IDLE.

### r65 DONE — CSA2 axis 3 (FP4) ACHIEVED. Store survives **4-bit, 8× smaller, no recall loss** (63 win 82.5=82.5,
1024→256 B/slot); 3-bit is a hard cliff. Keys are bit-limiting (matching needs 4-bit; values tolerate 3-bit).
Written up: r65/RESULTS, README. Committed. Both boxes IDLE.

### v0.6 AGENDA (the "actual memory" build, shared store / DeepSeek-CSA2 route; per-layer dropped as vanilla).
Plan: ~/.claude/plans/hey-lets-pin-down-giggly-dream.md. Rounds: r67 update/recency, r68 multi-hop, r69 learned
write gate, r70 collision@scale, r71 does-it-help-a-real-LM, later: thalamic relay (fetch multiple + gated), fuzzy recall.

### r67 DONE (box1) — update / recency WORKS. recency_off 88.9%, recency_on **96.7%** (82.3% @step500 vs off 42.2%),
nonupd control 97.5%. Position is NOT free recency (pos-free key) — the explicit slot_seq bias supplies it. Both
slots kept. Written up r67/RESULTS + README. First capability beyond lookup. Box1 idle.

### r68 DONE (box2) — multi-hop NEGATIVE. All arms ≈0% (retrieved 0.3, token/broken/residual 0.0); models never
fit 2-hop (loss 13.7->12.8, ppl ~1e6). Leading cause: a value_skip_norm value = scale·(norm(last)+norm(embed)) is
NOT a clean key (last-block half halves cosine to the intended key -> exact-match fails). Written up r68/RESULTS +
README. Diagnose in r69 (1-hop base? curriculum? long train?), then likely a re-projection into key space or the
thalamic relay. Box2 idle.

### r69 KILLED early (user pivot to multistore) — but SALVAGED the key datapoint: **1-hop in the hop-task
format WORKS** (box2 hop1_retrieved 98.8%, box1 hop1_token 95.5%). So the r68 negative is specifically 2-hop
CHAINING, not a format/warm-start bug — backs the "retrieved value isn't a clean exact key" diagnosis. See r69/RESULTS.

### r70 RETRACTED by r71 — the 85% was a task leak, NOT chaining. (Kept for record; retraction note in r70/RESULTS.)

### r71 DONE — REFUTES r70. Broken-chain control **85.5%** (= intact 85%, cutting the chain changed nothing ->
NOT chaining); seed-1 replication **7.8%** (collapsed to 0.5, did not reproduce) despite 97.3% 1-hop base. LEAK:
bpe_hop places the tail C as the last-written value with query always the head, so "output most-recent value" scores
~85% blind. **MULTI-HOP IS NOT SOLVED** — neither shared (r68) nor per-layer multistore (r70/r71) genuinely chains.
Written up r71/RESULTS, corrected r70/README. Box1/box2 idle.

### r72 DONE — leak-proof multi-hop: BOTH stores at CHANCE. per-layer 0.2% (broken 0.2%), shared 0.5% (broken 0.0).
Confirms r70 was the leak; neither topology does genuine composition. Cause: a raw retrieved value (scale·(norm(last)
+norm(embed))) isn't on the key manifold -> next hop's exact match never fires. Written up r72/RESULTS + README.
MULTI-HOP UNSOLVED as-is; needs a learned inter-layer key transform. Box1/box2 idle.

### r73 DONE — learned reindex NEGATIVE. reindex hop2 0.5% (seed0) / 0.2% (seed1, base 96.4%), = broken control
(0.2%). Linear map on the retrieved value didn't unlock chaining. Cause: hop_q detached + only current hop trained
-> reindex gets no signal for the intermediate key. Written up r73/RESULTS + README. Box1/box2 idle.

### r74 DONE — end-to-end gradient NEGATIVE. grad+reindex 0.5% (broken 0.3), grad-only 0.2% (broken 0.2). Chance
both arms. EXHAUSTS the implicit routes (r68/r70-71/r72/r73/r74). Feeding the raw retrieved value backward can't
work (off the key manifold). Written up r74/RESULTS + README. Box1/box2 idle.

### r75 DONE — pointer loop NEGATIVE (0.2%/0.0%, controls chance). CLOSES the implicit multi-hop line: r68 shared,
r70/71 per-layer leak, r72, r73 reindex, r74 e2e-grad, r75 pointer — ALL chance. Within-forward read-chaining is the
wrong organ. Written up r75/RESULTS + README. Box1/box2 idle.

### ===== NEW THRUST: memory as a managed WORKING MEMORY (user reframe) =====
Split faculties: context window = INPUT ONLY (transient); store = separate MEMORY (persistent working state = "how
current context is used"). Composition = read -> WRITE intermediate -> read across STEPS (working memory), not
read-read in one forward. Memory needs a CONTROLLER: a learned GATE picks the op per step — read-select / write-new
/ merge / update / skip — driven by novelty(max-sim) + agreement(value match). This is ALSO the space win
(merge/update/skip keep it O(facts) not O(tokens)).
PLAN (user-approved): (1) r76 build the gated-op controller on the PER-LAYER store (all layers see all facts;
simplest place to get the gate right). (2) r77+ transition to ONE store + per-layer ADAPTERS translating the single
content into each layer's space (store once = space-efficient; DeepSeek reindex generalized per-layer). Per-layer =
scaffold (N copies, temporary); one-store+adapters = destination.

### r76 DONE — managed-memory concept VIABLE, recall unsettled. Bounded ✓ (gate mem 198-251 vs nogate 508=full,
O(facts)). Updates ✓ (gate 51.2% vs nogate 7.8%, overwrite-in-place essential). Recall INCONCLUSIVE: gate 66.1% but
STILL CLIMBING at 3k trim (17->39->60->66) vs nogate 94.1% — confounded with under-training; possible prose-clobber
cap (a fact's key token recurs in prose -> overwrites its slot via value_shift). Perf: clone-once write_gated
(8x->0.64s/step). Written up r76/RESULTS + README. Box1/box2 idle.

### r77 DONE — MANAGED MEMORY IS VIABLE. gate_recall 86.1% @mem198 (STILL CLIMBING 66->83->86, no plateau -> the
gap to nogate 94% is UNDER-TRAINING, not the prose-clobber cap we feared); gate_update 85.3% @mem251 vs nogate 7.8%.
Bounded (~2.6x fewer slots, O(facts)) + recall-preserving (trending) + updates-land. CHECKPOINT MET ("if viable for
memory, good for next one store") -> one-store+adapters greenlit. Written up r77/RESULTS + README. Box1/box2 idle.

### r78 DONE — bounded gate reaches FULL recall, ZERO cost. At 12k: gate_recall **95.9%** @mem198 (>= nogate 94.1%
@mem508), gate_update **96.2%** @mem251 (vs nogate 7.8%). The 86->96 climb confirms the gap was under-training, not a
cap. Managed memory = bounded (2.6x fewer slots) + recall-preserving + updates-land; no salience gate needed. Written
up r78/RESULTS + README. GREENLIGHT one store. Box1/box2 idle.

### r79 DONE — ONE SHARED STORE SUFFICES; adapters NOT needed. shared+gate recall 96.2% / update 96.6% @mem198/251
= per-layer result (95.9/96.2) but ONE store not six (~6x fewer TOTAL slots). Per-layer read heads' out-projections
already translate the single content per layer. Managed bounded single store COMPLETE. Written up r79/RESULTS+README.

### ===== MILESTONE: managed, bounded, single shared store is done (r76-r79). Holds ≫ context window, sparse O(1)
read, 4-bit (r61-65); one slot per fact O(distinct keys), ~96% recall, updates land in place, one store not six. =====

### r80 KILLED (user: prioritise multi-hop) — but partial datapoint: one-store gated memory SCALES: f32 96.9% @mem229,
f64 96.1% @mem291 (bounded at 32/64 facts). Not written up (killed mid-run); noted here.

### ===== NORTH STAR: CoT OVER MEMORY (user). A model that reasons across facts held in its store (out of window).
Multi-hop is the test. Run BOTH routes; NEED ONE WORKING (user). =====
### ✅✅ ACHIEVED & CONFIRMED (r84, full 8k runs): CoT-OVER-MEMORY WORKS. Route A (token scratchpad) + write_skip_pad.
2-hop **96.4%** (seed0) / **96.9%** (seed1, best 97.0); 3-hop **95.2%** (each of 3 hops 98-99%: pos 98.8/97.7/98.8);
**BROKEN-CHAIN CONTROL 0.0%** (hop1 still 88.8%, hop2 **0.0%** -> cut the link and ONLY the 2nd hop dies = REAL
chaining, not the r70/71 return-last-value leak). The model emits each intermediate as a token then re-queries, each
hop a clean single-hop store lookup; composes over facts held OUT-OF-WINDOW in the bounded single store. r83 diagnosed
the cap (hop2=45%=0.5x90%, intermediate key-and-value collision); write_skip_pad (don't store pad-valued slots) removed
it order-independently (hop2 45%->98%). Route B (latent) stays dead (r82; 7th latent negative). Written up:
r84/HYPOTHESIS+RESULTS, README r83+r84, results JSONs pulled from both boxes. Committed.
NEXT: autoregressive eval (feed model's OWN emitted intermediate, not teacher-forced) = the final honest number
(expected ~= product of per-hop ~0.98 => ~0.96 @2hop); THEN the paper section. Memory stack now = managed bounded
single store (r76-79) + recency read (r67) + CoT token scratchpad (r84).
### r82 DONE (box2) — route B (loop-gate latent) NEGATIVE: 0.2%, broken 0.2%. Hard-decode loop doesn't chain. 7th
straight latent-multihop negative. Latent route PARKED (learned halt gate not pursued unless A exhausted). r82/RESULTS.
### r81 SUPERSEDED (box1) — route A had a MASKING BUG: marked query positions not answer positions -> eval scored a
copy (hop2 100%) / guess (hop1 0%), not retrieval. Found via the 0/100 split. Fixed (`ans` on answer token). Re-run=r83.
### r83 DONE — route A FIXED, retrieval REAL, but a STORE ARTIFACT caps hop2 at 45%=0.5x90%. Both hops are genuine
lookups (r81 was a masking bug). hop1 ~90% / hop2 ~45%: the intermediate t1 is both a value (t0->t1) and a key
(t1->t2); its spurious t1->pad slot clobbers the real t1->t2 slot 50/50 under the write-op gate. Mechanism sound,
store collision. Fixed in r84 by write_skip_pad (hop2 -> 98%). r83/RESULTS.
Code (A/B): bpe_chain task (explicit [q t_i t_{i+1}] blocks); loop_read/loop_iters/loop_layer; ReadHead split into
retrieve()+inject(); _pointer_key hard straight-through flag; HippoMemory.read loop.

Code landed r76: config write_op_gate; EpisodicStore.write_gated (clone-once) + value_tok; WriteHead routes + computes
value token; per-layer commit passes cur_tok; `mem N/M` occupancy log (HippoMemory.mean_occupancy).
LATER (after viability): merge (blend agreeing values) + LEARNED gate; then ONE store + per-layer ADAPTERS
(store once, translate per layer = space-efficient destination). MULTI-HOP reopens as r78: write the COMPUTED
intermediate across steps + read back (working-memory composition) — needs the managed store first.

### r66 DONE — curriculum is NOT required on real language (overturns r63). Both boxes idle.
- box1 ml + **curriculum**: f8 cold 59.4%, **f16 warm 100.0%**, **f32 warm 100.0%** (8k steps/rung).
- box2 ml **cold** (16k steps): **f16 cold 99.7%**, **f32 cold 98.0%** (best 98.1) — reached full recall COLD
  via a DELAYED PHASE TRANSITION (16f: 0.3%@2k → 8.8%@4k → 77%@6k → 98.9%@8k → 99.7%@16k; 32f follows one notch
  later). r63's "50% wall / 2nd embedding-competition bottleneck" was UNDER-TRAINING — it stopped inside the
  pre-transition plateau. Curriculum = training-efficiency (~half the steps), NOT a capability requirement.
- Written up: r66/RESULTS.md, README row. Paper corrected: placement.tex (rewrote "two bottlenecks" →
  "delayed transition"), discussion.tex ("curriculum is an optimisation, not a requirement"), conclusion.tex,
  language.tex (forward-ref nuance). Recompiles clean, 9pp.
- CRON NOTE: session-only cron (45c8648f) + local session sleep + GPU-box sleep = nothing advanced overnight.
  Loop only progresses while this session is awake AND boxes are awake.

### ✅ CSA2 FOR OUR STORE IS COMPLETE (r64+r65): cross-layer reuse (free) + candidate pool (token ~2 slots / ann
~M/12) + 4-bit KV. The store is a shared KV holding ≫ the context window, read in O(1) slots at 4 bits/element.
The v0.5 plan goal is met. Good stopping point — surface to the user.

### Open / next (all deferred; pick or user-steer):
(a) Real quantized-storage + O(1) index KERNELS — the probes MEASURE candidates/bits but still compute densely in
    fp32; true int4 storage + a token-hash/ANN kernel make the byte+compute wins literal (engineering, not research).
(b) Asymmetric keys-4/values-3 packing (measured to work in r65; not wired as a mode).
(c) Train WITH the pooled/quantized read (only if a gap appears; none did at eval).
(d) CSA2 axes combined on the harder BPE fact counts + curriculum (do they interact with the r63 2nd bottleneck?).

### Open / next (CSA2 is user-flagged FUTURE, don't auto-start):
(a) **truly sparse COMPUTE** (candidate pool, not full q·K scan) = CSA2 route 2 [[csa2-store-future-build]] — makes
the O(1)-slot read also O(1)-compute at large M. Big architectural build; user flagged for later.
(b) fold r59 into the objective: move single-layer placement to read+write@all (survives the cliff).
(c) sparse-read on the char-level / harder-key task (map where small-k breaks).

### (prev) NORTH-STAR RESULT — ESSENTIALLY DEMONSTRATED (r61, 2026-09-18). Store > context window AND query only
what's required, BOTH halves at once:
1. **Store > context window: DONE** (r51 — bridges ~M/128 windows; 31× at M=4096, 90%).
2. **Query only what's required: BUILT + shown.** Added `sparse` mode to mech/persist.py (read a filled big store
   with a top-k readout, flipping the frozen MemConfig readout at eval via object.__setattr__). **Smoke (M=1024,
   7-window store): top-1 read = top-8 = dense graded = 100%, touching only 1 of 889 slots.** The graded-trained
   model reads perfectly sparsely at k=1 — each query touches ONE slot out of the whole store.
- **r61 RUNNING (chained after r60 on box2):** confirms at scale — sparse-read sweep (top-k 1/2/4/8/dense) at
  7/15/31-window bridges (M 1024/2048/4096) on r50_bpe64. Expected headline: store holds 31 windows (3968 tok),
  each query touches 1/4096 slots, recall ~= dense.
- Route 2 (CSA2 cross-layer/candidate-pool) remains the efficiency generalization; this already lands the user's
  "1 result" on the read side.

### After r59 — TWO ROUTES (user, 2026-09-18):
1. **Diagnose the failure and target it** — whatever breaks (matching? content? bootstrapping/r25?), find the
   mechanism and attack it. slot@1 / read-mass / content-miss / gate diagnostics are already logged per run.
2. **Implement DeepSeek-V4.1-Flash / CSA2 into a SHARED store** — cross-layer KV reuse (few anchor layers compute,
   rest reuse), reused sparse candidate-pool selection, FP4-quantized values. See memory [[csa2-store-future-build]].
   Our store is external persistent shared-KV; r56 already showed cross-layer redundancy (sharing wins).

### (superseded, easy-regime) r57 unique-per-layer store vs shared (32-symbol):
everything, see breaking point"). Built `--per-layer-store` (each layer its own EpisodicStore, keyed+valued at that
depth; smoke-tested). box1: per-layer @ 8/16/32 facts + shuf. box2: shared symmetric @ 16/32/64. Chained after r56
phase2. Predict per-layer ≤ shared / breaks earlier (deep keys are contextual, not the exact layer-0 token key).

### (was) r56 RUNNING — WRITE-SIDE PLACEMENT ABLATION (finishing the user's msg 30/39 ask; read half was r41).
Added `--write-layers` (value = MEAN of hidden states at listed layers = write at every layer, same slot count;
plumbing in config/model/train, smoke-tested). MQAR r11 config (8f, M=512, cold write@last=100%). box1: `r56_wlast`
(write@5 repro) + `r56_wall` (write@0-5). box2: `r56_wall_shuf` (control) + `r56_wall_s1` (seed1). 6k steps each,
~35 min/box. Predict wall ≤ wlast (early states dilute the last-block value, r9). Write up when done.

### Other open items — disposition (reported to user):
- **Sleep/consolidation on the BPE/language store:** mechanism PROVEN on MQAR (r26-r33); NOT ported to BPE. Non-
  trivial (sleep re-encodes episodes from decoded transcripts; the BPE store's key/decode path differs) — a
  multi-round effort, not a quick finish. Scoped, not done.
- **tflite elastic grow/shrink store:** DEFERRED BY DESIGN (user "poc so we ball"; plan deferred bucketed tflite
  export). Store size is already a free runtime knob (buffers non-persistent, r51); on-device export is deployment
  engineering, not a research finding. Intentional deferral, not a gap.

### (superseded) r56 was going to be the write-side selective-storage lever; renumber that to r57 (the eviction
thread's write-side follow-on: threshold-based selective WRITE so prose is never stored). Still queued, lower prio.

### (orig r56 note) the write side (the one lever left from the eviction thread):
**Don't store the prose at all.** Either (a) a THRESHOLD-based selective write (commit a slot only if its surprisal
> τ) so the store holds facts + a tiny working set by construction — reaches the oracle at tight M; or (b) TRAIN
with `write_select="surprisal"` / a binding-aware write gate (r50's gate is near-open ~1.0) so the model learns to
read from a facts-only store. (a) needs write-path surgery (commit currently writes all S slots); (b) is a ~1hr
training round. Both boxes idle — launch r56 with fresh budget/attention; not forced this tick (guardrail: a fixed
write_frac can't keep all facts AND drop all prose, so the clean test needs threshold/training, not a quick sweep).

---
### (prev) SCALING SOLVED to 100%/32 facts; r50 DONE bar 1 rung; r51 = close on context-window claim

**r49 CLOSED:** BPE curriculum ladder = **100% at 8, 16, AND 32 facts** (real-language, out of context, from store;
32-from-8 in one jump). Scaling is curriculum-bound, not architecture-bound.

**r50 (numbers in hand, results/*.json local):** seed-2 ladder REPLICATES seed 1 — `bpe4_s2` cold **35.3%** (the
bootstrap rung), `bpe8_s2` warm **100%**, `bpe16_s2` holding **100%** at step 5k/12k (box1 still finishing the last
rung, ~30 min; final number owed but conclusive). `bpe64` warm-from-32 **99.1%** (best 99.2%) — the 512-slot store
does NOT bind at 64. `bpe64_shuf` scrambled control **0.16%** (chance) — no leakage at 64. **box2 is IDLE.**

### The result in one line
Real-language facts stated in prose, queried out of context, answered from the store: **8/16/32 facts at 100%,
64 at ~99%.** Char-level capped at 13 facts and ~80%; BPE single-token keys + curriculum removed both limits.

### NEW finish line (user, 2026-09-18): close the experiment on TWO things —
1. **Match/exceed a model's context window.** The model's attention window is `seg_len=128` tokens (segments do NOT
   cross-attend — the store is the only cross-segment channel). Demonstrate it answers from an EMPTY 128-token prompt
   about episodes spanning **K× that window** (K = written-span / seg_len). At `M=512` the store already holds ~4
   windows (508 slots ÷ 127/seg ≈ 4 segments = ~4× 128). Grow `M` → grow K. This is the "effective context ≫ literal
   window" headline the objective's statefulness claim needs, on real language.
2. **The general recipe for "store more."** Two independent axes: **span/tokens** scales LINEARLY with store size `M`
   (grow M, hold more episodes; checkpoints load across memory_size since store tensors are non-persistent buffers);
   **distinct facts** is curriculum-bound (bootstrapping), needs `M ≥ facts × slots-per-fact` but is NOT M-limited
   below that (r39). Single-token keys (BPE) remove the multi-token addressing tax (r36).

### r51 plan (offline, no training — uses existing 100% checkpoints; box2 free, box1 busy):
- **r51a persistence:** `mech.persist write|read` two-process save/reload on a BPE checkpoint (fill store, kill,
  reload in fresh process, answer empty prompt). `_restore_bpe()` ADDED to `mech/persist.py` this session (branches
  on `args.task=="bpe_recall"`; rebuilds the deterministic key/val word pools). NOT yet run/synced. Reloaded must ==
  in-process; `--mismatch` (wrong store file) must fall to chance.
- **r51b context-multiple sweep:** load a 100% checkpoint at growing `M` with matched `n_seg` so the store fills;
  measure recall vs K = written-span/128-window. Recall holding as K grows = "exceeds context by K×, bounded by M."
  Check tile semantics in `bpe_recall_batch` first (distinct-per-segment vs repeated facts) before claiming span.
- Checkpoints: box1 has `r50_bpe8_s2.pt` (100%, 8f), box2 has `r50_bpe64.pt` (99%, 64f) + `r49_bpe32.pt` (100%, 32f).
  Nothing is local — run on a box. Sync `mech/persist.py` to the box first.

### Other open: r50 16f final number + write-up owed; more seeds; consolidation/sleep on the BPE store (r26-r33 MQAR
only); where the store finally binds at very high fact counts (64 didn't).

### Ops: bracket-escape pgrep as `[m]ech.train` (the un-bracketed form self-matches the ssh command and reports a
phantom process — cost several kill cycles chasing a duplicate that was already dead). Otherwise: no trailer; sync
mech to BOTH boxes; verify BEFORE relaunching (a double-launch made an r49_bpe32 duplicate); mkdir logs first.

## Was (2026-09-17 ~20:30)

- **r33/r33b DONE and written up** into `r32/RESULTS.md`. Re-encode 6.3% -> **75.0%**, both fixes tying exactly
  (6 of 8 facts) despite 24 vs 95 slots written. Cause was a **double compression**: the decoded transcript already
  IS the compressed selection (round(0.25 x 127) x 3 = 96 = n_slots), so re-running the gate compressed it twice.
  The tie locates the residual 25% upstream of the policy, in decode fidelity. Not yet measured.
- **r34 memory arms RERUNNING after a configuration error of mine.** The first pass used
  `--read-layers 1 --write-layer 5` with no `--key-layer 0 --key-pos-free --query-init identity --freeze-query`,
  i.e. a layer-1 random-projection query against last-block keys — two spaces, a placement that has never worked
  even on MQAR. It gave a clean null (effect +0.003 vs 0.003 jitter, gate 0.191) that would have been written up as
  "memory does not help on natural text". **Always take the placement from r23/r30's run.sh, never from prose.**
  Kept as `r34_mem_misconfigured.json`; it is evidence for r35, not waste.
  - box2: `r34_mem`. box1: `r34_shuffle`, `r34_compress`. `r34_off` is memory-free and was NOT rerun.
  - Early read at step 1000 on the corrected run: gaps vs control s0 +0.008, s1 +0.017, s2 +0.016, s3 +0.015 —
    smaller at segment 0 than after it, which is the shape retrieval must have. Do not quote until 10k steps.
- Analysis: `uv run --group sleep python experiment/r34/analyse.py` (difference-in-differences, see r34/HYPOTHESIS.md).

## Was (2026-09-17 ~18:20)

- **box1 — r33 RUNNING** (`mech.sleep_evict`, tag `r33_reencode_fix`): does carrying the written slot positions with
  the episode repair the post-sleep re-encode? Baseline to beat is r32's **6.3%**; consolidation was already 100% and
  should stay there. Result lands at `experiment/r32/results/r33_reencode_fix.json`. Write into `r32/RESULTS.md`,
  which currently says the fix "is not implemented or measured".
- **box2 — r34 RUNNING** (`r34_off`, `r34_mem`): PG-19, the first natural-text round. 10k steps, ~45 min.
  Then run `ROOT=$HOME/... bash experiment/r34/run.sh controls` for `r34_shuffle` and `r34_compress` (box1, once free).
- Headline metric is the **per-segment loss profile**, printed each eval as `[s0:.. s1:.. s2:.. s3:..]`. `forward()`
  has no KV cache, so memory-off must be flat; a downward slope in memory-on is the store and cannot be bought with
  parameters (4.86M off vs 5.13M on — a 5.5% gap that absolute ppl cannot disentangle).
- **Watch the gate.** At step 60 `gates=1:0.000`. If it never opens, a flat profile is the r19-r25 bootstrapping
  deadlock, NOT evidence that memory cannot help natural text — and the escape is the known one (curriculum/warm
  start). Diagnose before concluding.

r31/r32 DONE and written up. r31: episode-aware eviction 0.5% -> 82.5%/90.6% under overflow, reproduced on seed 1
(ring 1.1% vs episode 100.0%); sizing rule is one episode of headroom. r32: compression and sleep interact in the
REPAIR path — a content-based write gate is a function of the weights, so after sleep it selects a different subset
and re-encoding collapses to 6.3%. That is what r33 fixes.

## Where the project stands

**Training-time (closed):** store size 1x -> 16x, fact count 8 -> 48, key vocabulary 32 -> 128 all reach 100% — and
all three failed cold for **one** reason, a gate/decoder bootstrapping deadlock that a curriculum escapes (r25).
Controls at chance at every size; the store persists across processes at 4x and 8x.

**Consolidation (open, r26-r27):** the store is a readable transcript (keys decode exactly, 100%, at 381 and 3,937
slots), so replay needs no decoder. The loop runs: consolidation 0% -> 100%, all 381 slots reclaimable, LM loss
improved by rehearsal. Open costs: new writes degrade to 75% (r28 tests the fix), drift > ~400 destroys the store, and
consolidation is well-posed only one user at a time.

## Next after r28

- If memory rehearsal restores new writes: measure the **steady-state loop** — repeated wake/sleep cycles, slots
  reclaimed per cycle vs slots filled per wake (throughput), and a retry budget for facts that never consolidate
  (livelock). That is the last thing the eviction design needs before it can be called done.
- If it does not: the write path degradation is structural; report it as the cost of consolidation and quantify how
  many sleep cycles the memory survives.

## Earlier STOPPED note (2026-09-15 ~14:10) — user: "kill"

r20 (relaunched 13:29 on box 1) KILLED by the user after the step-750 eval. Partial result: 8x store 96.2% @250,
100% @500, 100% @750 (slot@1 100%, mass 0.97). No checkpoint (--save writes at the end), no scrambled control, no
persistence test. Box processes gone; monitor stopped; cron deleted. Nothing running. Wait for the user.

To finish r20 properly later: rerun `experiment/r20/run.sh` (~75 min per run) — or cut to `--steps 750`.

(Earlier: box 1 = adhavan_23bce0839@10.30.161.19; the user said to use only this box.)

(History: r20 was killed by the user at 2026-09-14 21:35 before its first eval; the loop was paused overnight.)

## Was (21:30)

r20 was running from 21:25 (launched by the box waiter when r19 finished): warm start
(`--init experiment/r19/ckpt/r19_warm_skipnorm_M2048.pt`) at 8x (M=4096, 32 segments, batch 4 x accum 8, 1500 steps,
~1 h?), then the scrambled control. Monitor `monitor.sh r20`.

r19 DONE: warm-started 4x **100%**, scrambled 2.8%, persistence reloaded 100% / mismatch 2.5%. Written up.

## Next action when r20 is done

- Check step time early: if 1500 steps would take > ~1.5 h per run, note it, and let it run anyway.
- If 8x >= 95% and control at chance: persistence on its checkpoint (offline, `--batch 2`, since store files are ~4x
  larger), then write up. Next candidates: a second seed of the curriculum (1x -> 2x -> 4x on seed 1), or push capacity
  another way (more facts per store at 4x, e.g. 32 facts).
- If 8x starts near chance with read mass well below 0.97: the transferred read dilutes at 8x. Try a sharper read
  (`--beta 64`) before more curriculum.
- If it starts high and decays: rerun with `--lr 1e-4`.
- If the control is above chance: stop and diagnose.

## r7 -> r8 (done 23:36)

r7 stopped after `r7_w0s1`: taxonomy confirmed position hijack (query pos 126, reads at 101-126, 0% on fact
keys). r8 (`--key-pos-free`) still collapsed at step 750; stopped. Learned query steered away (identity
query 93.8% vs learned 0% on the collapsed ckpt). r9 = r8 + `--freeze-query`.

## r9 so far (00:05)

- `r9_w0s1` (block-0 value, frozen identity query, pos-free key): **100.0% on single-occurrence facts**, matching
  100% all run. H9b, H9e confirmed. First store to hold facts seen once.
- `r9_w5s1` (last-block value): chance through step 2500, gate shut (0.001), matching 100% -> content failure.
- r9_w5s1 final 2.3%, r9_w5s0 final 2.3% (both chance, matching 100% -> content). r9_w3s1 at chance through step 1000.
- **r10 CHAINED on the box** (1 waiter) — launches when r9 writes ROUND DONE. mech pushed (model.py b840ac43).
  **Do not launch r10 manually.** When r9 is done: write r9/RESULTS.md, arm `monitor.sh r10`, commit.
- r10 (was: ready, commit 8ffbb45):
  `--value-skip-layer 0` = value is last-block state + block-0 output (write stays at the last block).
  Push mech/ only after r9's ROUND DONE.

Suggested r10 if w5s1 ends at chance (sequential, single occurrence, 8 facts unless noted):
1. `w5_skip0` : --write-layer 5 --value-skip-layer 0 --value-shift 1   (objective placement + skip)
2. `w0s1_s1`  : r9_w0s1 config, seed 1
3. `w0s1_shuf`: r9_w0s1 config, --shuffle-store (control)
4. `w0s1_f16` : r9_w0s1 config, --n-facts 16
5. `w0s1_f32` : r9_w0s1 config, --n-facts 32
Then two-process persistence on the r9_w0s1 checkpoint (offline, CPU).

## Next action when r9 is done (r9 = r8's design + --freeze-query; read the notes below as r9)

Write `r9/RESULTS.md` (r7, r8 RESULTS done). Read r9 as:
- **First check H9e:** correct-slot reads must stay ~100% to the END of every run (r7 collapsed at step 1250).
- `w0s1` (value = block-0 output at v_j) is the ceiling. If it fails with matching intact, the read/value
  path is broken for single occurrences — fix before anything else.
- `w5s0` should fail (direct test of r6's prediction-caching diagnosis).
- `w5s1` working = "write at the last block" survives with observed content -> r10 scales it (16/32 facts
  single-occurrence, seeds 1-2, scrambled control, two-process persistence re-test).
- `w5s1` failing while `w0s1` works -> last block lost identity -> r10 tries a value skip path (value from
  block 0-3 while key stays at layer 0), or reports that the write position must move earlier.

## Established results (see OBJECTIVE.md, rN/RESULTS.md)

- 8 facts, tiled, key @0, value = last block shift 0: **100% on 3/3 seeds**; HC off 2.5%, scrambled 2.7%;
  store reloaded in a fresh process 100% == 100%, mismatched store 3.4%, foreign model refused.
- Limit: value = prediction; 16/32 facts (fewer repeats) fail on content with matching ~100%.
- Matching holds at 100% over a 1,920-slot store (r6b); recall there confounded by batch 8.

## Ops notes

- Sequential runs; 5 parallel max fit (2.77 GB each at B=32, M=512); parallel is slower per run.
- Memory scales with n_seg x M (every segment's [B,S,M] read tensors live until backward).
- Never rsync all of `experiment/` up during a round. Offline probes in `~/research/devtest` on CPU.
- Chain the next round on the box with an `until grep -q "ROUND DONE"` waiter when possible.
