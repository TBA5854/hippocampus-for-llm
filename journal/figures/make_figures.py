"""Figures for the journal manuscript, built from the experiment results.

Every plotted value is read from a result JSON or a synced run log under
experiment/. The two exceptions are constants taken from a committed
RESULTS.md / README row because the original log was overwritten when the two
boxes' logs were merged; each is marked SOURCE: README.

Palette: dataviz reference categorical slots 1-3 (validated all-pairs on a white
surface) + a 3-step ordinal blue ramp (validated --ordinal); controls in muted ink.
Identity never rests on colour alone: every series also has its own marker shape.

Nature Machine Intelligence figure rules applied here: <=180 mm wide; vector PDF with
embedded TrueType-42 text (Liberation Sans, a metric-compatible Arial); panel labels
8 pt bold lowercase, all other text 5-7 pt; axis lines with ticks; axes labelled with
units in parentheses; NO background gridlines; all text black (colour lives in the
marks and keys, not in text); 95% Wilson intervals as error bars where n is known.
Every plotted value is also written to journal/source_data/FigN.csv.

    python3 journal/figures/make_figures.py          # writes journal/figures/*.pdf
                                                     # and journal/source_data/*.csv
"""
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiment"
OUT = Path(__file__).resolve().parent
SRC = OUT.parent / "source_data"
PREVIEW = Path(sys.argv[1]) if len(sys.argv) > 1 else None

# ---- palette (dataviz reference, light) -----------------------------------------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ORD = ["#86b6ef", "#2a78d6", "#104281"]          # blue 250 / 450 / 650

_sans = "Liberation Sans" if any("Liberation Sans" in f.name for f in
                                 font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": _sans, "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.linewidth": 0.6, "lines.linewidth": 1.5,
    "lines.markersize": 5, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "text.color": INK, "legend.labelcolor": INK, "savefig.dpi": 300,
})
MM = 1 / 25.4
WIDTH = 180 * MM                                  # journal maximum
RING = dict(markeredgecolor="white", markeredgewidth=0.9)


def style(ax):
    """Open axes with tick marks; no background gridlines (NMI figure rule)."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", color=INK, pad=4)
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", color=INK, va="bottom")


def label(ax, x, y, text, color=INK, **kw):
    # all text is black: the journal asks for keys/keylines, not coloured text
    ax.annotate(text, (x, y), fontsize=6.3, color=INK, **kw)


def wilson(p_pct, n, z=1.959964):
    """95% Wilson score interval for a proportion given as percent; returns percent."""
    p = p_pct / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def ebars(ax, xs, ys, ns, color, **kw):
    """Draw 95% Wilson error bars for recall percentages ys measured on ns queries."""
    lo, hi = zip(*(wilson(y, n) for y, n in zip(ys, ns)))
    ax.errorbar(list(xs), list(ys), yerr=[[max(0.0, y - l) for y, l in zip(ys, lo)],
                                          [max(0.0, h - y) for y, h in zip(ys, hi)]],
                fmt="none", ecolor=color, elinewidth=0.6, capsize=1.5, capthick=0.6,
                zorder=kw.pop("zorder", 2), **kw)


SD = {}                                      # figure -> source-data rows


def sd(fig, panel_, series, x, y, n, scale=1.0, note=""):
    """Record one plotted point (y in percent, n queries) for the source-data file."""
    lo, hi = wilson(y, n) if n else (float("nan"),) * 2
    SD.setdefault(fig, []).append(dict(panel=panel_, series=series, x=x, y_percent=round(y, 3),
                                       n_queries=n, ci95_low=round(lo * scale, 3),
                                       ci95_high=round(hi * scale, 3), note=note))


def write_source_data():
    SRC.mkdir(exist_ok=True)
    for fig, rr in SD.items():
        with open(SRC / f"{fig}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rr[0]))
            w.writeheader()
            w.writerows(rr)


# ---- loaders ----------------------------------------------------------------------
def rows(path):
    return json.loads((EXP / path).read_text())["rows"]


def final(path):
    return json.loads((EXP / path).read_text())["history"][-1]["recall"] * 100


N_EVAL = 640     # every training-time eval is 20 batches x 32 queries (all 268 histories are
                 # multiples of 1/640); the store-swap probe is 32 queries (r86 swap.json)

LOGLINE = re.compile(r"^(top-\d+|graded\(all\))\s+store=\s*(\d+) slots \((\d+) windows\)"
                     r".*recall=\s*([\d.]+)%")


def logpts(path, read="top-1"):
    """(tokens bridged, recall %) for every matching line of a sparse-probe log."""
    pts = []
    for ln in (EXP / path).read_text().splitlines():
        m = LOGLINE.match(ln.strip())
        if m and m.group(1) == read:
            pts.append((int(m.group(3)) * 128, float(m.group(4))))
    return pts


def depth(path):
    r = rows(path)
    return [x["depth_frac"] for x in r], [x["recall"] * 100 for x in r], [x["n"] for x in r]


# ---- Figure 2: reach beyond the window + low-bit store ------------------------------
def fig2():
    from matplotlib.lines import Line2D
    # sample sizes: r101 JSON rows (n=48); r108 = batch 1 x 4 iters (n=4); r102 = batch 2 x 4
    # iters (n=8) per its RESULTS.md; r111 = batch 1 x 4 iters (n=4); the batch-2 scrambled
    # r111 scrambled control: batch 2 x --iters 4 = n=8 (command recorded in the session log).
    P = []   # (series key, x tokens, y %, n)

    def add(key, pts, n):
        for x, y in pts:
            P.append((key, x, y, n))

    # d512 / 2-layer (r87 ckpt): r101 grow store, <=25k tokens
    for ns in (16, 64, 128, 200):
        rr = rows(f"r101/results/grow_sparse_ns{ns}.json")
        top = next(r for r in rr if r["read"] == "top-1")
        den = next(r for r in rr if r["read"] == "graded(all)")
        P.append(("top1_d512", top["bridge_windows"] * 128, top["recall"] * 100, top["n"]))
        P.append(("dense_d512", den["bridge_windows"] * 128, den["recall"] * 100, den["n"]))
    for f in ("normal_ns1000", "normal_ns3000", "normal_ns7800"):   # r108
        add("top1_d512", logpts(f"r108/logs/{f}.log")[:1], 4)
    P.append(("dense_d512", 999 * 128, 0.0, 8))   # SOURCE: README r102 (d512 dense -> 0% at 127k)
    # d256 / 6-layer (r92 ckpt): r102 at 25k/127k, r111 to the million
    for f in ("normal_ns200", "normal_ns1000"):
        add("top1_d256", logpts(f"r102/logs/{f}.log"), 8)
        add("dense_d256", logpts(f"r102/logs/{f}.log", "graded(all)"), 8)
    add("top1_d256", logpts("r111/logs/box2_ns3000.log")[:1], 4)
    x7800 = rows("r111/results/box2_ns7800.json")[0]
    P.append(("top1_d256", x7800["bridge_windows"] * 128, x7800["recall"] * 100, x7800["n"]))
    # matched batch-2 scrambled-store control (d512), r111
    for f in ("ctrl2_shuffle_ns1000", "ctrl2_shuffle_ns3000"):
        add("scrambled", logpts(f"r111/logs/{f}.log"), 8)
    for key, x, y, n in P:
        sd("Fig2", "a", key, x, y, n)
    ser = lambda k: [(x, y, n) for kk, x, y, n in P if kk == k]

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 62 * MM),
                               gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.30))
    a.axvline(128, color=INK, lw=0.6, ls=(0, (3, 2)))
    label(a, 145, 24, "attention window\n(128 tokens)")
    spec = [("top1_d512", BLUE, "-o", dict(**RING), 3), ("top1_d256", BLUE, "-s",
            dict(mfc="none", mec=BLUE, mew=1.0, ms=6.5), 4),
            ("dense_d512", ORANGE, "-o", dict(**RING), 3), ("dense_d256", ORANGE, "-s",
            dict(mfc="none", mec=ORANGE, mew=1.0, ms=6.5), 4)]
    for key, col, fmt, kw, z in spec:
        p = ser(key)
        xs_, ys_, ns_ = zip(*p)
        ebars(a, xs_, ys_, ns_, col)
        a.plot(xs_, ys_, fmt, color=col, zorder=z, **kw)
    xs_, ys_, ns_ = zip(*ser("scrambled"))
    ebars(a, xs_, ys_, ns_, MUTED)
    a.plot(xs_, ys_, "X", color=MUTED, ms=5.5, mec="white", mew=0.6, zorder=5)
    label(a, 1.9e3, 104, "top-1 read: one slot touched")
    label(a, 1.9e3, 30, "dense read over\nall slots")
    label(a, 6.2e5, 80, "≈1M tokens\n(7,800× window)", ha="center")
    a.set_xscale("log")
    a.set_xlim(80, 2.2e6)
    a.set_ylim(-8, 112)
    a.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6], ["100", "1k", "10k", "100k", "1M"])
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_xlabel("distance between the fact and the query (tokens)")
    a.set_ylabel("recall (%)")
    a.legend(handles=[
        Line2D([], [], color=INK, marker="o", ls="", **RING, label="d512, 2 layers"),
        Line2D([], [], color=INK, marker="s", ls="", mfc="none", mec=INK, label="d256, 6 layers"),
        Line2D([], [], color=MUTED, marker="X", ls="", mec="white", label="scrambled-store control"),
    ], loc="lower left", bbox_to_anchor=(0.27, 0.03), frameon=False,
        handletextpad=0.3, borderaxespad=0.2)
    style(a)
    panel(a, "a", "Recall far beyond the attention window")

    # b: quantize the stored codes (r65 dense read, 31 windows / 3,937 slots, n=160 each).
    # "keys and values" is the 64-fact model, the keys/values ablation the 16-fact model,
    # so each series is shown relative to its own 16-bit (unquantized) recall.
    bits_order = [16, 8, 6, 4, 3, 2]
    series = [("r65/results/r65_both_M4096_n32.json", "keys + values", BLUE, "o"),
              ("r65/results/r65_keys_M4096_n32.json", "keys only", ORANGE, "s"),
              ("r65/results/r65_values_M4096_n32.json", "values only", AQUA, "^")]
    for path, name, col, mk in series:
        rr = {r["bits"]: (r["recall"] * 100, r["n"]) for r in rows(path)
              if r.get("pool", "none") == "none"}
        base = rr[16][0]
        x = [i for i, bt in enumerate(bits_order) if bt in rr]
        y = [100 * rr[bits_order[i]][0] / base for i in x]
        lo_hi = [wilson(rr[bits_order[i]][0], rr[bits_order[i]][1]) for i in x]
        b.errorbar(x, y, yerr=[[max(0.0, yy - 100 * lo / base) for yy, (lo, hi) in zip(y, lo_hi)],
                               [max(0.0, 100 * hi / base - yy) for yy, (lo, hi) in zip(y, lo_hi)]],
                   fmt="none", ecolor=col, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=2)
        kw = dict(mfc="none", mec=col, mew=1.0, ms=6) if mk != "o" else dict(RING)
        b.plot(x, y, "-" + mk, color=col, zorder=3, label=name, **kw)
        for i, yy, (lo, hi) in zip(x, y, lo_hi):
            sd("Fig2", "b", name, bits_order[i], yy, rr[bits_order[i]][1], scale=100 / base,
               note="recall relative to the 16-bit recall of the same model; CI = Wilson CI "
                    "of the quantized recall divided by the 16-bit recall")
    b.axvspan(2.6, 3.4, color=GRID, alpha=0.6, lw=0, zorder=0)
    label(b, 3.0, 110, "4 bits: no loss", ha="center")
    label(b, 3.58, 48, "keys fail\nat 3 bits")
    label(b, 4.3, 78, "values hold\nto 3 bits")
    b.set_xticks(range(len(bits_order)), [str(x) for x in bits_order])
    b.set_xlim(-0.3, 5.4)
    b.set_ylim(-8, 118)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_xlabel("precision of each stored element (bits)")
    b.set_ylabel("recall kept relative to 16-bit (%)")
    b.legend(loc="lower left", frameon=False, borderaxespad=0.2)
    style(b)
    panel(b, "b", "4 bits per element: no loss")
    return fig


# ---- Figure 3: passkey depth ------------------------------------------------------
def fig3():
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(WIDTH, 60 * MM), sharey=True,
                            gridspec_kw=dict(wspace=0.2))
    a, b, c = axs

    def draw(ax, fig_panel, name, path, fmt, col, zorder=3, **kw):
        x, y, n = depth(path)
        ebars(ax, x, y, n, col, zorder=zorder - 1, alpha=0.7)
        ax.plot(x, y, fmt, color=col, zorder=zorder, **kw)
        for xx, yy, nn in zip(x, y, n):
            sd("Fig3", fig_panel, name, xx, yy, nn)

    for i, (f, lab) in enumerate([("sat_ctx32", "4k tokens"), ("sat_ctx128", "16k"),
                                  ("sat_ctx512", "64k")]):
        draw(a, "a", f"d512, {lab} of context", f"r113/results/{f}.json", "-o", ORD[i], **RING)
        x, y, _ = depth(f"r113/results/{f}.json")
        label(a, x[-1] + 0.05, y[-1] + (6 if lab == "64k" else 0), lab, va="center")
    draw(a, "a", "d512, scrambled-store control", "r112/results/box1_ctx64_ctrl.json", "-X", MUTED,
         zorder=2, lw=0.9, ms=4.5, mec="white", mew=0.5)
    label(a, 1.05, -1, "scrambled\n(8k)", va="center")
    panel(a, "a", "Recall falls as the store fills")

    def mech(ax, prefix, title, letter):
        draw(ax, letter, "fixed store", f"r113/results/{prefix}sat_ctx512.json", "-o", BLUE, **RING)
        draw(ax, letter, "unique, always-novel keys", f"r113/results/{prefix}rarekeys_ctx512.json",
             "-s", ORANGE, **RING)
        draw(ax, letter, "append-only store", f"r113/results/{prefix}grow_ctx512.json", "^", AQUA,
             zorder=4, mfc="none", mec=AQUA, mew=1.0, ms=6)
        panel(ax, letter, title)

    mech(b, "", "Why it decays (d512, 64k tokens)", "b")
    label(b, 0.55, 64, "unique keys lift\nthe late tail 4×")
    b.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", **RING, label="fixed store"),
        Line2D([], [], color=ORANGE, marker="s", **RING, label="unique, always-novel keys"),
        Line2D([], [], color=AQUA, marker="^", ls="", mfc="none", mec=AQUA,
               label="append-only store\n(identical: not lapping)"),
    ], loc="upper right", frameon=False, borderaxespad=0.1)
    mech(c, "d256_", "Weaker model (d256, 64k tokens)", "c")
    label(c, 0.16, 55, "only the first-written fact\nsurvives; unique keys are\ntoo weakly embedded to help")
    for ax in axs:
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0\nfirst", "0.25", "0.5", "0.75", "1\nlast"])
        ax.set_xlim(-0.05, 1.2)
        ax.set_ylim(-8, 112)
        ax.set_xlabel("depth of the fact in the context (fraction)")
        style(ax)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("top-1 recall (%)")
    return fig


# ---- Figure 4: lookup table -> memory ------------------------------------------------
# panel order follows the order the main text discusses them: revision, bounded, composition
def fig4():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDTH, 60 * MM),
                                  gridspec_kw=dict(width_ratios=[0.9, 1.05, 1.25], wspace=0.42))

    def bars(ax, panel_, xs_, vals, cols, names, width):
        for x, v, col, nm in zip(xs_, vals, cols, names):
            ebars(ax, [x], [v], [N_EVAL], INK, zorder=2)
            ax.bar([x], [v], width=width, color=col, edgecolor="white", linewidth=1.0, zorder=3)
            sd("Fig4", panel_, nm, x, v, N_EVAL)
            label(ax, x, wilson(v, N_EVAL)[1] + 2.5, f"{v:.1f}", ha="center")

    # a: revision — a re-stated fact must return its newest value (r67)
    vals = [final("r67/results/r67_recency_off.json"), final("r67/results/r67_recency_on.json"),
            final("r67/results/r67_nonupd.json")]
    bars(a, "a", range(3), vals, [ORD[0], BLUE, MUTED],
         ["recency read off", "recency read on", "control: not updated"], 0.55)
    a.set_xticks(range(3), ["off", "on", "control:\nnot updated"])
    a.set_xlabel("recency-biased read")
    a.set_ylim(0, 118)
    a.set_yticks([0, 25, 50, 75, 100])
    a.set_ylabel("newest value recalled (%)")
    style(a)
    panel(a, "a", "Revising a re-stated fact")

    # b: bounded store — gated new/update/skip ops (r76 ungated vs r78 gated, converged)
    ung = [final("r76/results/r76_nogate_recall.json"), final("r76/results/r76_nogate_update.json")]
    gat = [final("r78/results/r78_gate_recall_12k.json"), final("r78/results/r78_gate_update_12k.json")]
    w = 0.34
    xu = [i - w / 2 - 0.02 for i in range(2)]
    xg = [i + w / 2 + 0.02 for i in range(2)]
    nm = ["recall", "update in place"]
    bars(b, "b", xu, ung, [ORD[0]] * 2, [f"append every occurrence (508 slots): {n}" for n in nm], w)
    bars(b, "b", xg, gat, [BLUE] * 2, [f"gated ops, one slot per fact (~200 slots): {n}" for n in nm], w)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=ORD[0], label="append every occurrence (508 slots)"),
                      Patch(color=BLUE, label="gated ops, one slot per fact (≈200 slots)")],
             loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    b.set_xticks(range(2), nm)
    b.set_ylim(0, 142)
    b.set_yticks([0, 25, 50, 75, 100])
    b.set_ylabel("accuracy (%)")
    style(b)
    panel(b, "b", "A bounded store")

    # c: multi-hop through the store (r93/r94 teacher-forced, r95 autoregressive)
    tf = {2: "r93/results/r93_chain_2hop_f64.json", 3: "r94/results/r94_chain_3hop_f64.json",
          4: "r93/results/r93_chain_4hop.json", 5: "r94/results/r94_chain_5hop.json"}
    ar = {2: "ar_2hop_f64", 3: "ar_3hop_f64", 4: "ar_4hop", 5: "ar_5hop"}
    hops = [2, 3, 4, 5]
    ytf = [final(tf[h]) for h in hops]
    arj = [json.loads((EXP / f"r95/results/{ar[h]}.json").read_text()) for h in hops]
    yar = [j["final_autoregressive"] * 100 for j in arj]
    nar = [j["n"] for j in arj]
    ybr = [final(tf[h].replace(".json", "_broken.json")) for h in hops]
    c.plot(hops, [100 * 0.98 ** h for h in hops], "-", color=MUTED, lw=0.8, zorder=1)
    label(c, 3.1, 76, "grey line: 0.98ⁿ (98% per hop)")
    ebars(c, hops, ytf, [N_EVAL] * 4, BLUE)
    c.plot(hops, ytf, "-o", color=BLUE, **RING, zorder=3, label="teacher-forced")
    ebars(c, hops, yar, nar, ORANGE)
    c.plot(hops, yar, "s", color=ORANGE, mfc="none", mec=ORANGE, mew=1.0, ms=6.5, zorder=4,
           label="autoregressive (own intermediates)")
    ebars(c, hops, ybr, [N_EVAL] * 4, MUTED)
    c.plot(hops, ybr, "-X", color=MUTED, lw=0.9, ms=5, mec="white", mew=0.5, zorder=2,
           label="broken-chain control")
    for h, y1, y2, y3, n2 in zip(hops, ytf, yar, ybr, nar):
        sd("Fig4", "c", "teacher-forced", h, y1, N_EVAL)
        sd("Fig4", "c", "autoregressive", h, y2, n2)
        sd("Fig4", "c", "broken-chain control", h, y3, N_EVAL)
    for h, y in ((2, ytf[0]), (5, ytf[-1])):
        label(c, h, y + 6.5, f"{y:.1f}%", ha="center")
    c.set_xticks(hops)
    c.set_xlim(1.7, 5.3)
    c.set_ylim(-8, 118)
    c.set_yticks([0, 25, 50, 75, 100])
    c.set_xlabel("chain length (hops through the store)")
    c.set_ylabel("joint recall (%)")
    c.legend(loc="center left", bbox_to_anchor=(0.0, 0.45), frameon=False)
    style(c)
    panel(c, "c", "Reasoning across facts out of the window")
    return fig


def main():
    for name, fn in (("fig2_longcontext", fig2), ("fig3_passkey", fig3), ("fig4_memory", fig4)):
        fig = fn()
        fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
        if PREVIEW:
            PREVIEW.mkdir(parents=True, exist_ok=True)
            fig.savefig(PREVIEW / f"{name}.png", bbox_inches="tight", dpi=220)
        plt.close(fig)
        print("wrote", name)
    write_source_data()
    print("wrote source data:", ", ".join(f"{k}.csv ({len(v)} rows)" for k, v in SD.items()))


if __name__ == "__main__":
    main()
