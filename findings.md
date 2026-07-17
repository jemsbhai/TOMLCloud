# Findings

## Curated Summary

[To be populated as EXP-001 and subsequent analyses complete. Numbers cited in
the CloudCom 2026 revision must trace to entries in the Raw Findings Log below
and to archived artifacts in data/phase6_attribution/.]

---

## Raw Findings Log

Chronological, append-only. Corrections are recorded as dated errata entries.

### 2026-07-17 -- Repo audit: v0 attribution analysis used four approximations

**Key result:** The Table VI numbers in the IC2E submission (proportional mean
70.1%, TOML mean 35.7%) were produced by `scripts/run_phase6_analysis.py` (v0)
under four approximations, all correctable offline from archived data.

**Evidence:**
1. E_total: `e_total_est = max(e_a_coloc, e_b_coloc)` (v0, attribution loop)
   instead of integrating telemetry; host dmon traces in
   `data/phase5_colocation/host_*.csv` were collected but never read.
2. TED: no temperature adjustment applied; Eq. 7 in the paper specifies
   E_per_TO(T) and Section V-E concedes the reported numbers omit it.
3. Co-located TO counts: proxied from Phase 2 isolated `batches_completed`;
   Phase 5 summaries record actual co-located counts, which differ sharply
   (RNN: 75,430 isolated vs 29,852 co-located with VGG-16, a 60% throughput
   loss; VGG-16: 3,456 vs 3,159, an 8.6% loss).
4. Idle: hardcoded `idle_power = 44.0` in the Phase 3 recompute path; every
   run's summary contains its measured `idle_power_w` (observed 47.7-51.2 W).

**Significance:** The submitted headline comparison (70.1% vs 35.7%) is
directionally supported by the data but numerically provisional. EXP-001
supersedes it with a corrected pipeline and a four-rung ablation
(proportional, TOML-raw, TOML-static, TOML+TED). Revised paper reports only
EXP-001 outputs.

**Additional provenance notes from the same audit:**
- Repetition counts by phase: Phases 1, 2, 4 have 3 runs per configuration;
  Phase 3 is one 15-minute run per architecture; Phase 5 is one run per pair
  direction. The paper's blanket "3 repetitions per configuration" claim must
  be scoped accordingly in the revision.
- Phase 5 summaries record mean_temp_c, batches_completed, to_total, and
  idle_power_w per container, which is sufficient for TED-adjusted attribution
  with no new experiments.
- Host traces show a startup stagger (e.g., RNN active alone at ~107 W before
  VGG-16 joins at ~390-406 W), enabling a concurrent-window diagnostic.
- Paper text inconsistency found during audit: abstract claims proportional
  allocation "misattributes container energy by up to 7.7x" but 7.7x is the
  power-draw range across architectures; measured worst-case misattribution in
  Table VI is 190.8%. To be unified in revision.
- Citation error found during audit: Section VI-B cites "Cloud Carbon
  Footprint [11]" but [11] is Dodge et al. (FAccT 2022), a different work, not
  the Cloud Carbon Footprint tool. To be corrected in revision.

### 2026-07-17 -- EXP-001: Corrected attribution ablation (headline results)

**Key result:** Under the corrected pipeline, proportional allocation mean
attribution error is 70.2% (max 194.9%); conservative TOML+TED is 32.9%
(max 77.7%). The corrected numbers confirm the submitted headline comparison
(70.1% vs 35.7%) directionally and quantitatively for the baseline.

**Evidence:** data/phase6_attribution/ablation_summary.csv,
attribution_comparison.csv, ablation_results.json; LOGBOOK.md EXP-001.
E_total independently corroborated by host dmon traces within -3.4% to
-5.3% on all six pairs (host_crosscheck.csv).

**Significance:** These are the numbers the CloudCom 2026 revision reports.
Key structural findings for the paper: (1) toml_raw error is identically the
measured MPS throughput loss (mean 31.3%; resnet18 retains only 20.0% of
isolated throughput beside vgg16, lstm 19.8% beside resnet50), so residual
TOML+TED deviation from the isolated reference is ~1.6 points beyond
physically real slowdown; (2) TED gains concentrate where the
calibration-to-colocation temperature delta is large (coefficient
adjustments up to +38.7%); (3) per-container telemetry overstates
light-tenant energy by 5.7x to 17.8x under MPS (contamination factor),
the sharpened CTI headline; (4) lstm+gru shows negative interference
overhead: consolidation saves 30.6% energy for ~16% throughput cost each,
and conservative attribution credits the saving correctly.

### 2026-07-17 -- ERRATUM: Submitted Table III (TED%) superseded

**Key result:** Submitted Table III TED% values (4.2 to 21.5, with stated
range "6.8% (VGG-16) to 21.5% (ResNet-18)") cannot be reproduced from the
archived Phase 3 data under either the measured or the hardcoded idle
assumption; the computation path that produced them was not archived.
Recomputed values (measured idle, per-window regression, r2 reported):
resnet18 9.32 (r2 0.968), resnet50 9.78 (0.941), vgg16 2.32 (0.554),
rnn 6.00 (0.874), lstm 14.59 (0.993), gru 6.15 (0.905),
xgboost 8.66 (0.314).

**Evidence:** data/phase6_attribution/ted_recomputed.csv; LOGBOOK.md
EXP-001 Results.

**Significance:** Revised paper regenerates Table III from
ted_recomputed.csv, rewrites the dependent narrative (new range 2.3 to
14.6; LSTM is now the maximum; ResNet-50 leakage-isolation observation
retained with the new value 9.78%), and replaces the abstract/intro claim
of "7 to 21 percent" thermal attribution error. The xgboost slope (r2
0.314) is caveated wherever its adjustment is used. SPP recompute matched
submitted Table IV to three decimals, so Table IV stands; this erratum is
confined to Table III and text derived from it.

### 2026-07-17 -- ERRATUM: Run-1 values standardized to means of 3 in Table II and Table VII

**Key result:** The submitted paper's quoted power range (45.5W to 348.0W), the
Table II VGG-16 power entry (348W), and the Table VII A100 E/TO column used
run-1 values while surrounding captions claimed means of 3. Verified against
all 21 Phase 1 summaries: Table II bare-metal energy column IS means of 3
(honest); the power figures and Table VII were run-1. Standardized to means:
power range 45.2W (XGBoost) to 347.4W (VGG-16), ratio 7.68 (still 7.7x);
Table VII A100 E/TO means 6.55, 8.93, 4.89, 7.49, 5.09, 5.71 (x1e-15 J) with
ratios 1.49, 1.15, 1.07, 1.53, 0.92, 0.85. The five-of-six within 1.5x claim
survives (RNN at 1.53 remains the exception).

**Evidence:** data/phase1_baremetal/phase1_baremetal/*_run{1,2,3}_summary.json;
hand-verified means recorded here; regenerated by scripts/generate_figures_v1.py.

**Significance:** Revision reports mean-based values throughout; figures
regenerated on the same basis so tables, figures, and prose agree.

### 2026-07-17 -- Submitted to CloudCom 2026 (IEEE, Paris, Oct 27-29)

**Key result:** The revised paper was submitted to the CloudCom 2026 main
track via EasyChair on 2026-07-17, 29 days ahead of the August 15 deadline.
Double-blind; anonymized author block, acknowledgments commented out, no
repository link (code available on request during review, open source on
publication). Final length within the 8-page IEEE limit.

**Evidence:** paper/tomlcloud_ic2e2026.tex (submitted state),
paper/references.bib (19 entries, all DOIs verified against live sources);
all reported numbers trace to data/phase6_attribution/ artifacts (EXP-001).

**Significance:** Closes the IC2E revision cycle. Each IC2E objection has a
structural answer in the submitted version:
- R1 "metrics never enter the model": new Section III-D defines the
  integrated attribution model before evaluation; TED enters the
  coefficients, SPP justifies TO-share residual distribution, CTI quantifies
  the residual and telemetry contamination.
- R1 "no ablation": four-rung ablation (proportional / TOML raw / TOML static
  / TOML+TED) on identical inputs, plus per-container throughput retention.
- R1 "TOML rests on an unknown venue paper": published FLAIRS-39 paper cited
  directly, with a TOML-establishing passage in the introduction.
- R1 "only MPS evaluated": explicit sharing-mode scoping argument in
  Section II-B (time-slicing is temporal accounting; MIG power apportionment
  is itself unsolved, cited); MPS is where attribution is hardest.
- R2 "practical value not justified": quantified billing distortion in
  Discussion (RNN pays 2.9x attributed energy; VGG-16 undercharged 49.5%).
- R2 "idle power policy unclear": explicit delta-energy convention and
  platform-fixed-cost policy in Section IV-E.
- R2 "remaining error still high": decomposition shows mean 32.9% deviation
  vs 31.3% measured throughput loss, i.e. ~1.6 points attributable to the
  model.
- R2/R3 "Table II overlapped by Figure 2": all floats moved to [t]; Fig. 2
  removed as duplicative of Table II.
- R3 "Kepler mischaracterized": corrected to counter-based regression models
  with eBPF attribution on CPU, utilization-ratio division on GPU; upgraded
  to the peer-reviewed IEEE CLOUD 2023 citation.
- R3 "figures never referenced": all remaining figures cited in prose at the
  point of discussion; unreadable Fig. 3 removed, replaced by a TED
  regression figure showing the exact quantity in Table III.
- R3 "no code link": availability statement added (anonymized form for
  review).
- R3 "11 citations": bibliography expanded to 19, 12 peer-reviewed.
- R3 "TED range contradicts Table III": superseded by the EXP-001 recompute
  (erratum above); Table III now reports r2 per fit.

**Open items for camera-ready (if accepted):**
- Restore author block and acknowledgments (marked RESTORE AT CAMERA-READY).
- Replace availability clause with https://github.com/jemsbhai/TOMLCloud and
  make the repository public.
- Add FLAIRS DOI to references.bib; confirm published title punctuation.
- Confirm El Ahmad surname indexing and Yang et al. author list.
- Consider broadening the LLM-assistance disclosure to reflect analysis and
  writing collaboration in this revision.
