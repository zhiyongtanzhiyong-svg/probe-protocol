# Submission Checklist — Computers and Electronics in Agriculture

> **Source:** Drafted 2026-10-13. ScienceDirect/Elsevier guidelines page returned 403 to all
> automated fetch attempts (webfetch, curl, Jina, CDP). Items marked **[VERIFY]** must be
> confirmed against the live "Guide for Authors" page before submission:
> `https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture/publish-your-article/guide-for-authors`

## STATUS (2026-10-14)
**Re-target to C&E in Agriculture (Elsevier) COMPLETE:**
- [x] Document class MDPI → `\documentclass{article}` (plain LaTeX); all MDPI macros removed
- [x] Citations author-year → numbered `[1]…[25]` (verified in compiled PDF)
- [x] End-matter MDPI macros → standard `\section*{...}` (Author Contributions / Funding / Data Availability Statement / Conflicts of Interest)
- [x] `\simplesumm` removed; Highlights created (`Highlights.txt`, 5 bullets, 50–200 chars each)
- [x] Compiles clean (pdflatex ×2, 0 errors, 0 undefined refs); checker 189/189; 0 TABs
- [x] K=1 LAION p-value corrected to 1.5×10⁻⁵ throughout (abstract + intro + body)
- [x] LP++ DFN draw `30.5/48.0/56.4` + full gain `+6.5` pp; Appendix-A "Feature-store unification" (LAION 75.4% vs canonical 76.1%, no change of sign)
- [x] **2026-10-14 A+B expansion (reframed):** Lever B = honest readout-geometry characterization (checkpoint-dependent CLS-vs-patch-token signal; DFN→CLS, LAION→patches; ties Shi et al. 2026; "not a new method") — contributions (5), main body, untested-heads line, Limitation (11)
- [x] Cross-crop now **tomato + corn** (both powered); seed-paired powered stats added (tomato K1 +4.3pp p=0.051; corn K1 +8.1pp p=0.021, K10 p<0.001); Highlights bullet 5 updated to two cross-crops
- [x] **Full 55k PlantVillage skipped** (user decision) — tomato+corn already provide powered multi-crop evidence
- [x] M5 gate green: 189/189 checker PASS; clean compile 27pp, 0 errors, 0 undefined refs/cites, 0 overfull>100pt, 0 TABs
 - [x] **2026-10-16 adversarial review pass + apt fix.** Cross-examination flagged P1/P2/P3; each checked against source: **P2 (K=1) already addressed** (abstract L25 "checkpoint-specific… probe-protocol effect… not a representation effect"); **P1 restricted-usage-mode already in abstract** (L30) and title L12 already tobacco-scoped/representation-focused → no retitle needed. **P3 is a research gap** (fine-tuned CLIP not run at K≥5), already in Limitation L553 — not a writing fix. **Sole apt residual fixed:** abstract L26 "…the ordering is a property of the data regime rather than of the model" conditioned on the frozen-probe-vs-fine-tuned-CNN protocol (→ "…rather than of the encoder"), aligning the abstract with the body's own L482 "restricted usage mode" scoping. Re-verified: 189/189 gate + clean compile 27pp (0 errors / 0 undef / 0 overfull>100pt).
 - [ ] **Still TODO before submission:** replace repository URL placeholder; add author ORCID; confirm journal caps (keywords, page/word limit) against the live Guide for Authors; (optional) consider a 5th highlight for the readout-geometry finding if the journal allows 5

## Abstract & Keywords

- [x] Abstract unstructured (no sub-headings) — **done** (238 words, measured 2026-10-24)
- [x] Abstract ≤ 250 words — **done** (238 words, measured 2026-10-24; it had drifted to ~575 words through the revision rounds and was re-cut during the reviewer-response pass, with the removed detail already present in the Introduction and Results) **[VERIFY: confirm exact limit]**
- [x] Highlights ≤ 85 characters each — **done** (all five were 85–95 characters; re-cut 2026-10-24, now 76–83) **[VERIFY against the live Guide for Authors]**
- [x] Keywords: 7 (CLIP; zero-shot classification; agricultural disease diagnosis; vision-language models; few-shot learning; knowledge graph; prompt engineering) **[VERIFY: confirm C&E-in-Ag keyword range — some Elsevier journals cap at 3–6; may need to trim to 6]**
- [x] No `\simplesumm` (MDPI-specific) — **removed**

## Citation & References

- [x] **Citation style:** Now **numbered** `[1], [2], …` — **done** (article class + `thebibliography` renders `\cite` as `[n]`; verified `[1]…[25]` in compiled PDF)
- [x] **25 references — no orphans** — **pass** (25 distinct `\cite` keys = 25 `\bibitem` entries; parity clean)
- [x] **Numbering follows first-citation order (strict)** — **pass** (`[1]` CLIP … `[8]` Shi et al. … `[25]` DINOv2; `\bibitem` blocks ordered by first `\cite` appearance). Managed by `scripts/audit_references.py --reorder` (committed, idempotent); verified `order_strict=True`.
- [x] **Reproducible reference audit — pass via `scripts/audit_references.py`** (each `\bibitem` resolved against Crossref/DataCite/arXiv/HF; title + first-author + year checked). `tld` (community HF dataset, no DOI) verified **offline against local ground truth** (`data/tld_local_manifest.json`: 1,392 images / 10 classes / 0 missing on disk). Audit record: `data/reference_audit.json`.
- [x] DOI present for most references — **pass** (4 without DOI: `radford2021learning`, `dinov2`, `fang2023data`, `tld` — all arXiv/preprint or community dataset with no publisher DOI available)

## Section Structure

- [x] IMRAD: Introduction → Related Work → Materials and Methods → Results → Discussion → Conclusions — **present**
- [x] Results organised by RQ1/RQ2/RQ3 — **present**
- [x] Appendix A (reproducibility protocol) — **present**
- [x] Abbreviations section — **present** **[VERIFY: confirm C&E in Ag requires abbreviations section]**

## End-Matter (required by journal-guidelines skill)

- [x] CRediT authorship contribution statement — **added** (4 authors: Lizhuo Zhang, Feier Hao, Chenfeng Long, Xiaoyong Tang; roles L.Z./F.H./C.L./X.T. — **VERIFY role assignment with co-authors**)
- [x] Declaration of competing interest — **added**
- [x] Acknowledgments (with AI-use declaration) — **added**
- [x] Data availability statement — **added** (repository URL placeholder)
- [ ] ORCID for authors — **MISSING** **[VERIFY: confirm required; add ORCID iDs for all four authors]**
- [ ] Repository URL placeholder — **MUST replace before submission**

## Figures & Tables

- [x] Figures in PDF vector format (`\includegraphics{figures/fig_*.pdf}`) — **pass**
- [x] Tables use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`) — **pass**
- [x] Captions above figures, below tables — **pass**
- [ ] **[VERIFY: confirm minimum DPI (typically 300 for halftones, 1000 for line art)]**
- [ ] **[VERIFY: confirm figure placement rules (embedded in text vs. separate files)]**
- [ ] **[VERIFY: confirm color requirements (color OK? grayscale required for B&W section?)]**

## Document Class & Layout

- [x] **Document class:** Switched MDPI → `\documentclass{article}` (plain LaTeX) — **done**
  - All MDPI-specific macros removed: `\firstpage`, `\pubvolume`, `\issuenum`, `\articlenumber`, `\pubyear`, `\copyrightyear`, `\datereceived`, `\daterevised`, `\dateaccepted`, `\datepublished`, `\AuthorNames`, `\address`, `\corres`, `\reftitle`, `\Title`, `\Author`, `\simplesumm`, `\authorcontributions`, `\funding`, `\dataavailability`, `\conflictsofinterest`
  - Note: `elsarticle.cls` not installed locally, so plain `article` is used; re-wrap in the official `elsarticle` class at final typesetting if the portal requires it
- [ ] **[VERIFY: confirm page limit (typically no hard limit for C&E in Ag, but ~15–25 pages is typical)]**
- [ ] **[VERIFY: confirm line spacing (single vs. 1.5 vs. double)]**
- [ ] **[VERIFY: confirm font (Times New Roman vs. Computer Modern)]**
- [ ] **[VERIFY: confirm single-column vs. double-column]**

## Blinding

- [ ] **[VERIFY: confirm single-blind (author names visible to reviewers) vs. double-blind]**
  - If single-blind: current author name/affiliation is fine
  - If double-blind: must remove author name, affiliation, ORCID, acknowledgments from manuscript

## Cover Letter

- [ ] Draft cover letter with:
  - Significance statement (why this paper matters for C&E in Ag readership)
  - Confirmation that the paper is not under consideration elsewhere
  - Suggested referees (3–5, with ORCIDs and affiliations)
  - Statement of AI use in manuscript preparation
  - **[VERIFY: confirm C&E in Ag requires suggested referees]**

## Submission Portal

- [ ] **[VERIFY: confirm submission portal URL (likely Editorial Manager at `submit.elsevier.com` or `editorialmanager.com`)]**
- [ ] **[VERIFY: confirm accepted file formats (.tex, .docx, .pdf)]**
- [ ] **[VERIFY: confirm APC (C&E in Ag is hybrid — confirm whether OA is optional or required)]**
- [ ] **[VERIFY: confirm word/page limit for full papers]**
- [ ] **[VERIFY: confirm supplementary material policy (size limit, format)]**
- [ ] **[VERIFY: confirm replication package / code deposit requirements]**

## Pre-Submission Actions

- [ ] Replace `[ANONYMOUS_REPOSITORY_URL_TO_BE_INSERTED_BEFORE_SUBMISSION]` with actual URL
- [ ] Add author ORCID
- [x] Switch document class to Elsevier template — **done** (article class)
- [x] Convert citation style to numbered (if required) — **done** ([1]…[25])
- [x] Remove all MDPI-specific macros — **done**
- [x] Re-compile and verify 0 errors, 0 undefined references — **done** (pdflatex ×2, 0 errors)
- [x] Run `scripts/check_tex_vs_data.py` — confirm 189/189 — **done** (189/189)
- [x] Create Highlights (`Highlights.txt`) — **done** (5 bullets)
- [x] Draft cover letter — **done** (`cover_letter_CeA.md`)
- [ ] Proofread for grammar and style (use `latex-paper-en` grammar module)
- [ ] Verify all figure files exist and are referenced correctly
