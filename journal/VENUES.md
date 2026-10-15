# Venue scoping — "Memory as a Block"

Scoped 2026-10-07. Impact factors are the latest Clarivate JCR (released June 2026, 2025
citation data) unless noted. The journal-format manuscript goes in this folder once a venue
is picked — the format depends on the venue.

## Bottom line

| goal | venue | why |
|---|---|---|
| **Highest IF that fits** | Nature Machine Intelligence (23.9) | best conceptual fit for "stateless weights, stateful store" — but needs scale, baselines, and a 3,500-word rewrite first |
| **High IF, long-form OK** | IEEE TPAMI (20.4) / IEEE TNNLS (9.7) | take long papers; both expect head-to-head baselines |
| **Best fit as the paper stands** | TMLR (no JCR IF) or JMLR (6.8) | rigorous long-form with honest negatives is what they reward; both feed the NeurIPS/ICLR/ICML Journal-to-Conference track |

## Ranked by impact factor

| venue | IF (5-yr) | fit for this paper | format | notes |
|---|---|---|---|---|
| Nature Machine Intelligence | **23.9** (31.8) | strong on concept, weak on evidence-at-scale | Article: ≤3,500 words main text, ≤150-word abstract, ≤6 display items; Intro / Results / Discussion / Methods | high desk-reject rate; **does not accept presubmission enquiries**; chronicle + round log → Supplementary |
| IEEE TPAMI | **20.4** | moderate — vision-heavy history, expects exhaustive baselines | long papers OK | slow review |
| Artificial Intelligence Review (Springer) | **18.8** (19.1) | weak for the main paper — scope includes "research reports", but it publishes mostly surveys, which is what drives the IF | — | fully open access since 2024, APC US$3,390; a natural home for a *survey* spin-off on memory-augmented transformers |
| Nature Communications | **18.1** | weak–moderate — must lead with the broad / hippocampal-memory angle | Nature-family format | fully open access, high APC |
| TACL | **11.7** | moderate if framed as long-context LM memory | short page limit — would need heavy cutting | presented at an *ACL conference |
| Patterns (Cell Press) | **10.8** | weak–moderate | — | data-science scope |
| IEEE TNNLS | **9.7** (11.9) | good — memory-augmented NN architectures are core scope | long papers OK | more attainable bar than TPAMI |
| Neural Networks (Elsevier) | ~6–7 | good — Hopfield / neuro-inspired memory | long papers OK | sources disagree (6.3 vs 7.2); check JCR |
| JMLR | **6.8** | very good — rigorous, no page limit | no page limit | free, open access; prestige in ML far above its IF; J2C-eligible |
| Artificial Intelligence (AIJ) | 4.7 | moderate | — | broad AI |
| Neural Computation (MIT) | 2.1 | strong topical fit (associative memory, consolidation) | — | low IF |
| TMLR | none in JCR (Scopus-indexed) | **excellent as-is** — judges correctness and evidence, not hype | no page limit | fast turnaround by design; J2C certification → present at NeurIPS / ICLR / ICML |

## The ML caveat on impact factor

In ML, journal IF is a weak signal — NeurIPS, ICML and ICLR carry more weight than almost any
journal. The **Journal-to-Conference (J2C) track** bridges this: papers published in JMLR or TMLR
(TMLR needs a J2C, Featured, or Outstanding certification) since 1 Jan 2025, within a 2-year
window, can be presented at NeurIPS, ICLR or ICML. The 2026 request deadlines have passed
(ICLR 25 Jan, ICML 3 May, NeurIPS 26 Sep 2026); the 2027 cycle should follow the same pattern —
confirm on the J2C page. Net: a TMLR/JMLR paper can get both a journal record and a top-conference
slot.

## What any top venue will ask for (gaps, from our own results)

1. **Scale.** Every result is a 7.8–12.1M-param model (d256/6L = 7.8M, d512/2L = 12.1M). One ~100M–1B result showing
   the mechanism holds is the single biggest lever — essential for Nature MI and TPAMI.
2. **Baselines.** No head-to-head against Memformer, RMT, Memorizing Transformers, Transformer-XL
   or RAG on the same recall / long-context tasks. Related work positions against them; reviewers
   will want numbers.
3. **Standard benchmarks.** Passkey is done (r112/r113). BABILong and RULER are not.
4. **Efficiency is measured, not literal.** The 4-bit and ~2-slot pool results have no hashed /
   int4 kernel behind them (already stated in Limitations).
5. **Format.** 29 pp with a 113-round chronicle suits JMLR / TMLR unchanged. For the Nature family
   it becomes ≤3,500 words of main text with the chronicle and round log in Supplementary.
6. **Front matter.** Author is still "TBA" and the affiliation is empty.

## Paths (pick one for this work — no dual submission)

- **A — maximize impact factor.** Close gaps 1–3 → full submission to Nature Machine Intelligence (it takes no presubmission enquiries). Requirements: [NMI_GUIDELINES.md](NMI_GUIDELINES.md). Fallback order: TPAMI, then TNNLS.
- **B — fastest credible publication.** Submit to TMLR now (fits as-is), request J2C
  certification → present at ICLR / ICML / NeurIPS 2027. JMLR is the slower, higher-IF variant
  of the same route.
- Either way, post the arXiv preprint first to timestamp the work; Nature Portfolio journals,
  JMLR and TMLR all accept preprinted submissions.

## Sources

- Nature Machine Intelligence IF — https://www.journalmetrics.org/journal/nature-machine-intelligence-2
- Nature MI article format — https://www.nature.com/natmachintell/content
- IEEE TPAMI IF — https://www.bioxbio.com/journal/IEEE-T-PATTERN-ANAL
- Artificial Intelligence Review IF — https://researcher.life/journal/artificial-intelligence-review/3894 ; APC — https://link.springer.com/journal/10462/how-to-publish-with-us
- Nature Communications IF — https://impactfactorforjournal.com/nature-communications-impact-factor/
- TACL IF — https://en.wikipedia.org/wiki/Transactions_of_the_Association_for_Computational_Linguistics
- Patterns IF — https://www.cell.com/metrics
- IEEE TNNLS IF — https://researcher.life/journal/ieee-transactions-on-neural-networks-and-learning-systems/2391
- Neural Networks IF — https://www.researcher.life/journal/neural-networks/4076
- JMLR IF — https://www.journalmetrics.org/journal/journal-of-machine-learning-research
- AIJ IF — https://en.wikipedia.org/wiki/Artificial_Intelligence_(journal)
- Neural Computation IF — https://www.journalmetrics.org/journal/neural-computation
- TMLR — https://jmlr.org/tmlr/ ; ISSN record https://portal.issn.org/resource/ISSN/2835-8856
- J2C track — https://iclr.cc/public/JournalToConference
