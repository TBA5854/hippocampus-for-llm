# Nature Machine Intelligence — formatting, writing and policy requirements

Collected 2026-10-12 from the journal's own pages (sources at the end). Everything below
applies to an **Article**. A compliance check of `main.tex` is at the bottom.

## 1. Article at a glance

| item | limit / rule |
|---|---|
| Main text | **≤3,500 words**, excluding abstract, Methods, references and figure legends |
| Abstract | **≤150 words**, unreferenced |
| Display items | **≤6** figures and/or tables |
| Structure | Introduction (**no heading**) → Results → Discussion → Methods |
| Headings | Results and Methods use topical subheadings; **the Discussion has none** |
| References | **~50** recommended as a guideline |
| Extended Data | optional, **≤10** display items, online only, not copy-edited by the journal |
| Supplementary Information | optional; sent to reviewers |

## 2. Before submitting

- **Fit.** The scope explicitly lists machine learning, deep learning, artificial neural networks,
  natural language processing, neuro-inspired computing and AI and cognitive science.
- **No presubmission enquiries.** NMI does not accept them; contact the journal directly if needed.
- **Preprints are encouraged** at any time and are not prior publication. Disclose the preprint
  DOI and licence on submission, and update the preprint record with the published DOI later.
- **Conference proceedings** are allowed only if the submission is a substantial extension; cite
  the proceedings paper.
- **No overlapping submissions.** Content must not significantly overlap any paper of yours under
  consideration or in press elsewhere (conference abstracts excepted). If you submit a related
  manuscript elsewhere during review, send NMI a copy. **The long `paper/` cannot go to TMLR in
  parallel.**
- **Publishing model.** Subscription (no charge), or Gold Open Access with an APC of
  **US$12,850 / £9,390 / €10,850**.
- **Peer review** is single-anonymized by default. Double-anonymized review is optional: you
  anonymize the manuscript yourself and tick the box, and the editor does not check it.

## 3. Initial submission

- **Format is free.** No special formatting is needed. PDF, Word or LaTeX are accepted; LaTeX
  goes in as a **compiled PDF**.
- **Components:** the manuscript file (including Methods, figures and any Extended Data), a cover
  letter, and optional Supplementary Information.
- **The manuscript must contain:**
  - all authors' names and affiliations (in the cover letter instead, under double-anonymized
    review);
  - enough method detail for an expert to replicate the work;
  - a reference list.
- **The cover letter** (reviewers don't see it) must:
  - explain the importance of the work and why it suits NMI's diverse readership;
  - disclose related manuscripts under consideration or in press elsewhere;
  - say whether you have discussed the work with an NMI editor;
  - optionally suggest reviewers, or name exclusions with reasons.
- **What gets sent to review.** The editor decides on the degree of advance, the soundness of
  the conclusions, how far the evidence supports them, and their wide relevance to the
  readership. Papers judged of "specialist interest", lacking novelty or conceptual advance, or
  with major technical problems are rejected without review.

## 4. Writing and language

- The paper must be accessible to non-specialists. Copy editors edit substantially after
  acceptance for exactly this reason.
- Avoid jargon, and explain it where it can't be avoided.
- Keep abbreviations to a minimum and define each at first use.
- Explain the background, rationale and main conclusions clearly.
- The title and abstract must be understandable by any scientist.
- No paper is rejected for poor language.

## 5. Required statements and sections

- **Methods:** short **bold** headings; specific subsections such as **Statistics** are
  encouraged. Methods-only references continue the numbering.
- **AI use must be declared.** LLMs cannot be authors, and LLM use must be documented in the
  Methods. Springer Nature's tiered policy:
  - **Green** (permitted): language polishing, suggesting structure.
  - **Amber** (permitted only with human oversight, verification and **disclosure**):
    suggesting analytical or experimental approaches, drafting summaries, comparing results with
    the literature, extensive writing support.
  - **Red** (not permitted): generating hypotheses, analyses or conclusions *and presenting them
    as human-derived*; using an LLM to generate core research reasoning **without disclosure**;
    assigning accountability to an AI.
- **Data availability** (mandatory): access to the "minimum dataset" needed to interpret, verify
  and extend the work, preferably in a repository (Zenodo, Figshare, Dryad), with datasets cited
  in the reference list.
- **Code availability** (mandatory when custom code is central): a **separate section after Data
  availability and before the references**, saying how the code can be accessed. The code must
  be shared with editors and reviewers **during review**, since NMI peer-reviews central code;
  Code Ocean is an optional free service for this. At publication, deposit it in a DOI-minting
  repository (Zenodo or Code Ocean), cite it, and preferably use an OSI-approved licence.
- **Software submission checklist.** Requested before review for central new code. The form is
  an Adobe-only dynamic PDF that couldn't be read as text; Nature's software policy asks for a
  README covering:
  - system requirements;
  - an installation guide;
  - a demo on small data, with expected output and run time;
  - instructions for use and for reproduction.

  Open the form in Adobe Reader to confirm the exact fields.
- **Acknowledgements:** brief; don't thank referees or editors; no dedications.
- **Funding statement:** a separate section, e.g. "A.B.C. discloses support for the research of
  this work from Funder [grant number xxxx]… G.H.I. declares no relevant funding."
- **Author contributions:** required, including for a single author. The corresponding author
  must provide an **ORCID**.
- **Competing interests:** one standard sentence, "The authors declare no competing interests."
  or "The authors declare the following competing interests: …".

## 6. References

- **Numbered, in this order:** main text → Methods → Data availability → Tables → Figure legends
  → Extended Data.
- One publication per number; footnotes are not supported.
- **Titles of cited articles are required** for Articles. Format:
  `Eigler, D. M. & Schweizer, E. K. Positioning single atoms with a scanning tunnelling microscope. Nature 344, 524–526 (1990).`
- Preprints: `… Preprint at https://arxiv.org/abs/… (year).`
- Datasets and code with DOIs go in the list (authors, title, repository, DOI as a URL). Web-site
  URLs go in parentheses in the text, not in the list.
- Work under review without a preprint is mentioned in the text, not cited. Grants are never
  references.

## 7. Figures (journal requirements + Nature figure guide)

- **Width:** at most **180 mm**. Resolution at least 300 dpi (450 is the proof maximum). Supply
  vector **.pdf or .eps** with **editable text**, in RGB.
- **Fonts:** standard sans-serif (Arial or Helvetica), embedded (TrueType 2 or 42), not
  outlined. Use the Symbol font for Greek letters.
- **Text size:** **panel labels 8 pt bold, upright, lowercase a, b, c**; all other text **5–7 pt**.
- **Required:** axis lines with tick marks; axes labelled with units in parentheses ("recall
  (%)"); an accessible, colour-blind-safe palette (e.g. Wong 2011).
- **Avoid:**
  - **background gridlines**;
  - decorative icons, drop shadows and patterns;
  - text over busy backgrounds and overlapping text;
  - **coloured text** (use keys or keylines with black text).
- Error bars where appropriate. Don't flatten labels onto images.
- **Legends:**
  - a brief title, then each panel described in sequence;
  - avoid methodological detail;
  - describe keys verbally (e.g. "open red triangles"), not with symbols;
  - state centre values, what the error bars are and how they were calculated, *n*, the
    statistical test and P values.
- **Citation:** cite every figure in sequence in the text ("Fig. 1", "Fig. 2"); panels in order.
- **Cover images:** generative-AI images are not permitted.

## 8. Tables, equations, Supplementary Information, source data

- **Tables** go at the end of the text document. Statistical tables state their error analysis
  in the legend.
- **Equations** go in the main text, numbered "(1)" and referred to as "equation (1)".
- **Supplementary Information:**
  - Label each item Supplementary Note, Discussion, Equation, Figure, Table, Video, Data or
    Software.
  - Number every type except Notes sequentially, separately from the main and Extended Data
    numbering.
  - **Cite every item at least once in the main text or Methods, using the word
    "Supplementary".**
  - Submit as **one combined PDF**; large tables can go as .xlsx or .csv, software as .zip.
  - Each Supplementary Figure must fit on one page with its legend.
  - It is published as supplied.
- **Source data** are encouraged: statistics source data in Excel, one file per figure.

## 9. Final submission after Acceptance in Principle (LaTeX)

- Word or LaTeX only; **PDF is not accepted** at this stage.
- Use a standard class (article, revtex or amsart), with **all text in one file** in **default
  Computer Modern**.
- **No non-standard fonts or packages; remove all personal macros.** Use graphicx for graphics.
- **Paste the .bbl into the .tex**, and delete `\bibliography` and `\bibliographystyle`.
- It must compile **with no errors or warnings**. House style is applied automatically.
- The Springer Nature LaTeX template is available.
- The corresponding author links their ORCID before final acceptance.

## 10. Compliance check of `journal/main.tex` (updated 2026-10-12, after the fixes)

Legend: ✅ done · 🟡 done, needs the author's input (red `[PENDING]` marker in the PDF) ·
⏳ waits for the larger-scale runs.

| requirement | status | how it is met |
|---|---|---|
| ≤3,500-word main text, ≤150-word abstract, ≤6 display items, ≤~50 refs | ✅ ~2,270 / 146 / 6 / 31 | counted by script, pending placeholders excluded |
| No intro heading; Results subheadings; no Discussion subheadings | ✅ | — |
| Title and abstract understandable by any scientist; jargon defined at first use | ✅ | abstract rewritten; context window, token, key–value cache, residual stream, Hopfield network, top-1, quantization, passkey, chain of thought, PG-19 each glossed where first used |
| AI-use disclosure in Methods | 🟡 | "Use of AI tools" paragraph; the sentence on human oversight is the author's to confirm |
| Statistics subsection; error bars and intervals; *n* in every legend | ✅ | "Statistics and reproducibility" in Methods; 95% Wilson intervals on Figs 2–4 and in Table 1; *n* stated per figure; no tests used and said so |
| Data availability, then Code availability, each its own section before the references | 🟡 | sections exist; Zenodo DOI, repository URL, licence, reviewer link and README still pending |
| Dataset and code DOIs in the reference list | 🟡 | PG-19 is cited; the Zenodo entries are added once the DOIs exist |
| Acknowledgements, Funding, Author contributions, Competing interests | 🟡 | all four sections exist with markers |
| Author name, affiliation, ORCID, correspondence | 🟡 | title block placeholders |
| Figures ≤180 mm, vector PDF, embedded TrueType-42 / Type-1 sans text | ✅ | Figs 2–4 are 151–155 mm; Fig. 1 is a standalone Helvetica PDF, 140 mm |
| Panel labels 8 pt bold lowercase; other text 5–7 pt | ✅ | set in `make_figures.py` and `fig1_architecture.tex` |
| No background gridlines; no coloured text; axes with ticks; units in parentheses; CVD-safe palette | ✅ | gridlines removed, all text black, colour carried by marks and keys |
| Figures and panels cited in sequence | ✅ | Fig. 1a, 1b (both cited in the introduction), Table 1, Fig. 2a/b, 3a/b/c, 4a/b/c (panels reordered to match the text) |
| Legends: title, panel-by-panel, keys described verbally, centre values, error bars, *n*, test | ✅ | rewritten |
| SI items labelled and each cited in sequence with the word "Supplementary" | ✅ | Supplementary Notes 1–9 and Supplementary Tables 1–6, plus Supplementary Data 1 (the round log as CSV), first cited in exactly that order |
| SI as one combined PDF | ✅ | `supplementary.pdf` (13 pp), with the round log as a separate CSV, `supplementary_data_1_experiment_log.csv` |
| Source data per figure | ✅ | `source_data/Fig2.csv`, `Fig3.csv`, `Fig4.csv`, `Table1.csv` (CSV; convert to .xlsx if the system asks) |
| References: numbered, titles present, preprints as "Preprint at URL (year)" | ✅ | checked in the compiled list |
| Cover letter | 🟡 | `cover_letter.md`; placeholders for preprint, reviewers, prior editor contact |
| Software submission checklist and code README | ⏳🟡 | the Adobe-only form must be opened in Adobe Reader; the README (requirements, install, small demo with expected output and run time, reproduce) is not written yet |
| Scaling and baselines subsection, Fig. 5 | ⏳ | `\pending{}` placeholders, wait for the box |
| Final-stage LaTeX (Computer Modern, no personal macros, inline .bbl, no warnings) | 🟡 mostly | `mathptmx`, `tikz` and the `\recall` macro are already gone and Fig. 1 is a file; at Acceptance in Principle only inline the .bbl, delete `\bibliography` and `\bibliographystyle`, and drop `lineno` and `\pending` once the placeholders are filled |

## Sources

- Submission guidelines (index): https://www.nature.com/natmachintell/submission-guidelines
- Content types: https://www.nature.com/natmachintell/content
- Preparing your material: https://www.nature.com/natmachintell/submission-guidelines/preparing-your-submission
- Formatting your initial submission: https://www.nature.com/natmachintell/submission-guidelines/initial-formatting
- Writing and language: https://www.nature.com/natmachintell/submission-guidelines/writing-and-language
- AIP and formatting (final): https://www.nature.com/natmachintell/submission-guidelines/aip-and-formatting
- Presubmission enquiries: https://www.nature.com/natmachintell/submission-guidelines/presubmission-enquiries
- Double-anonymized peer review: https://www.nature.com/natmachintell/submission-guidelines/dapr
- Editorial process: https://www.nature.com/natmachintell/submission-guidelines/editorial-process
- Publishing options: https://www.nature.com/natmachintell/submission-guidelines/publishing-options
- Aims & scope: https://www.nature.com/natmachintell/about/aims
- Editorial policies: https://www.nature.com/natmachintell/editorial-policies
- Preprints & conference proceedings: https://www.nature.com/natmachintell/editorial-policies/preprints-conference-proceedings
- Peer review policy: https://www.nature.com/natmachintell/editorial-policies/peer-review
- AI policy: https://www.nature.com/nature-portfolio/editorial-policies/ai
- Reporting standards (data, code): https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards
- Authorship: https://www.nature.com/nature-portfolio/editorial-policies/authorship
- Competing interests: https://www.nature.com/nature-portfolio/editorial-policies/competing-interests
- Figure specifications: https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/
