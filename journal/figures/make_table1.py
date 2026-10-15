"""Source data and 95% Wilson intervals for Table 1 of the journal manuscript.

Counts k of n queries come from result JSONs where one exists; the three that have no
JSON are marked SOURCE and were read from the run's own RESULTS.md / log line (the value
printed there is k/n to the precision shown, which this script asserts).

    python3 journal/figures/make_table1.py     # writes journal/source_data/Table1.csv
"""
import csv
import json
from pathlib import Path

from make_figures import EXP, SRC, wilson


def hist(path, step=None):
    h = json.loads((EXP / path).read_text())["history"]
    return (h[-1] if step is None else next(x for x in h if x["step"] == step))["recall"]


def k_of(p, n, shown=None):
    k = round(p * n)
    assert abs(k / n - p) < 1e-9, (p, n)
    if shown is not None:
        assert round(100 * k / n, 1) == shown, (k, n, shown)
    return k


swap = json.loads((EXP / "r86/results/swap.json").read_text())
R = []   # (experiment, condition, k, n, source)
for nf, f in ((8, "r50/results/r50_bpe8_s2.json"), (16, "r49/results/r49_bpe16_curr.json"),
              (32, "r49/results/r49_bpe32_curr.json"), (64, "r50/results/r50_bpe64.json")):
    R.append(("single-token keys", f"{nf} facts", k_of(hist(f), 640), 640, f))
R.append(("single-token keys", "scrambled store (64 facts)",
          k_of(hist("r50/results/r50_bpe64_shuf.json"), 640), 640, "r50/results/r50_bpe64_shuf.json"))
R.append(("persistence", "store reloaded in a fresh process", 160, 160, "SOURCE: r51/RESULTS.md (100.0%)"))
R.append(("persistence", "mismatched store file", 0, 160, "SOURCE: r51/RESULTS.md (0.0%)"))
R.append(("placement", "mean of all layers, one slot per token", 517, 640,
          "SOURCE: r103 log line 'recall 80.8%' (517/640)"))
R.append(("placement", "every layer writes its own slot, seed 0", 611, 640, "SOURCE: r103 (95.5% = 611/640)"))
R.append(("placement", "every layer writes its own slot, seed 1", 633, 640, "SOURCE: r103 (98.9% = 633/640)"))
R.append(("placement", "same slots keyed by the layer's residual (control)", 1, 640, "SOURCE: r103 (0.2% = 1/640)"))
for key, cond in (("A_store1_query1_own", "store A loaded, A's facts"),
                  ("A_store2_query2_own", "store B loaded, B's facts"),
                  ("A_store2_query1_swap", "store B loaded, A's facts"),
                  ("A_store1_query2_swap", "store A loaded, B's facts"),
                  ("A_store1_query1_scrambled", "scrambled store")):
    R.append(("store swap (one frozen model)", cond, k_of(swap[key], 32), 32, "r86/results/swap.json"))
R.append(("cross-model", "model 1 on its own store", k_of(hist("r86/results/r86_modelA.json"), 640), 640,
          "r86/results/r86_modelA.json"))
R.append(("cross-model", "model 2 on its own store", k_of(hist("r86/results/r86_modelB.json"), 640), 640,
          "r86/results/r86_modelB.json"))
R.append(("cross-model", "model 2 reads model 1's store", k_of(swap["B_store1_query1_foreign"], 32), 32,
          "r86/results/swap.json"))
R.append(("cross-model", "model 1 reads model 2's store", k_of(swap["A_storeB_queryB_foreign"], 32), 32,
          "r86/results/swap.json"))

SRC.mkdir(exist_ok=True)
with open(SRC / "Table1.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "condition", "k_correct", "n_queries", "recall_percent",
                "ci95_low", "ci95_high", "source"])
    for e, c, k, n, s in R:
        lo, hi = wilson(100 * k / n, n)
        w.writerow([e, c, k, n, round(100 * k / n, 2), round(lo, 2), round(hi, 2), s])
        print(f"{e:30s} {c:55s} {100*k/n:6.2f}  n={n:3d}  [{lo:5.1f}, {hi:5.1f}]")
