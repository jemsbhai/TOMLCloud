# TOMLCloud Experimental Logbook

Append-only. Corrections are dated addenda, never edits to past entries.

---

## EXP-001: Corrected attribution pipeline and ablation ladder (offline re-derivation)

**Date:** 2026-07-17 (planned); completed 2026-07-17
**Researcher:** Muntaser Syed (analysis script authored collaboratively with Claude)
**Type:** computational (offline analysis, no GPU)
**Status:** completed

### Hypothesis

The attribution numbers reported in the IC2E submission (Table VI: proportional
mean 70.1%, TOML mean 35.7%) were produced by `run_phase6_analysis.py` (v0)
under four approximations: (1) E_total taken as max of the two container-side
delta energies instead of an integration over telemetry, (2) no TED adjustment
of calibrated E_per_TO despite Eq. 7 specifying it, (3) co-located TO counts
proxied from isolated batch counts instead of the recorded co-located counts,
(4) hardcoded 44 W idle instead of per-run measured idle baselines.

We hypothesize that correcting (1)-(4) and adding TED-adjusted coefficients
(sensitivity from Phase 3, level from Phase 2, applied at the co-located
temperature recorded in Phase 5) will reduce mean TOML attribution error
relative to the corrected TOML-static variant. The proportional baseline error
is expected to remain the same order of magnitude (its 50/50 split does not
depend on coefficients, only on the corrected E_total). All reported numbers
in the revised paper will be superseded by this experiment's outputs,
whatever they turn out to be.

### Independent Variables

- Attribution method (4 headline levels, evaluated on identical inputs):
  1. `prop` - proportional 50/50 by active duration (cAdvisor-equivalent, Eq. 6)
  2. `toml_raw` - TO_count x E_per_TO(calibrated), no residual distribution
  3. `toml_static` - Eq. 7 with static calibrated coefficients (overhead by TO share)
  4. `toml_ted` - Eq. 7 with TED-adjusted coefficients (overhead by TO share)
- Secondary variant recorded but not headline: `toml_ted_eqsplit`
  (overhead split equally instead of by TO share).

### Dependent Variables / Metrics

- Absolute percentage attribution error per container per pair, vs ground truth
- Signed percentage error (over/under-attribution direction)
- Aggregates per method: mean, median, max over all 12 container observations

### Ground Truth Definition (stated explicitly for the paper)

E_true(A_i) = mean Phase 2 isolated delta energy (idle-excluded) over 3 runs.
Interpretation: the fair-billing reference, i.e. what A_i's 60-second GPU
occupancy costs when run alone. Residual error under TOML methods legitimately
includes contention-induced throughput loss (the container completed less work
in the same wall time under MPS); this is surfaced, not hidden.

Idle-power policy (answers IC2E R2): all energies are delta energies above the
per-run measured idle baseline. Idle power is treated as a platform fixed cost,
not attributed to tenants. The revised paper states this policy explicitly.

### Control Conditions

- Same six co-location pairs, same Phase 5 raw data, same E_total, same ground
  truth across all four methods. Only the attribution rule varies.
- TED sensitivity (slope) comes exclusively from Phase 3; coefficient level
  comes exclusively from Phase 2; applied temperature comes exclusively from
  Phase 5. No quantity derived from the evaluation target enters the model
  (no circularity).

### Protocol

1. Aggregate Phase 2 per-architecture: mean delta_energy_j, mean_temp_c
   (calibration temperature T_cal), mean energy_per_to, CV of delta energy.
2. Fit Phase 3 TED regression per architecture: per-window E_per_TO
   (run-average throughput assumption, measured idle) vs mean window
   temperature. Record slope (J/TO/degC), r2, TED%.
3. For each pair: merge both containers' 50 ms power samples by timestamp into
   100 ms bins; E_total = sum over union window of (bin mean power - mean
   measured idle) x 0.1 s. Record stagger and concurrent-window fraction.
4. Independent cross-check: integrate the host nvidia-smi dmon trace
   (active region above idle) and record percent agreement with step 3.
5. Compute the four attribution methods; errors vs ground truth.
6. CTI decomposition per container per pair: raw CTI (telemetry-contaminated),
   co-located vs isolated throughput ratio, temperature delta, TED coefficient
   adjustment %, contamination factor (reported delta energy / TED-attributed
   energy).
7. Recompute SPP from Phase 4 summaries and TED% from Phase 3 as consistency
   checks against submitted Tables III and IV.
8. Archive all outputs to data/phase6_attribution/ with run metadata and
   SHA-256 hashes of every input summary consumed.

### Environment

- **Hardware:** local Windows laptop (i9-14900HX, 64 GB RAM); no GPU required
- **Software:** Python 3.x, numpy, scipy (versions recorded in run_metadata.json)
- **Script:** scripts/run_phase6_analysis_v1.py (v0 preserved untouched)
- **Git commit:** to be recorded when committed
- **Seeds:** N/A (fully deterministic)

### Results

Host cross-check: PASSED. All six pairs agree within -3.4% to -5.3%
(host consistently lower; 1 Hz integer-watt sampling and active-region edge
trimming account for the systematic direction). E_total validated by
independent telemetry.

Ablation summary (12 container observations, abs error vs E_iso ground truth):

| method       | mean%  | median% | max%   |
|--------------|--------|---------|--------|
| prop         | 70.20  | 47.44   | 194.92 |
| toml_raw     | 31.29  | 15.75   | 80.23  |
| toml_static  | 34.90  | 30.58   | 79.85  |
| toml_ted     | 32.89  | 30.56   | 77.70  |
| ted_eqsplit  | 34.08  | 30.71   | 82.33  | (secondary)

Corrected proportional baseline (70.20%) reproduces the submitted 70.1%
almost exactly; corrected toml_static (34.90%) is close to the submitted
35.7%. The submitted headline comparison survives the pipeline corrections.

Identity confirmed: toml_raw per-container error equals measured throughput
loss exactly (raw = TO_coloc x E/TO_cal, E_iso = TO_iso x E/TO_cal, so
raw/E_iso = TO_coloc/TO_iso). Mean throughput loss across observations:
31.3%. Extremes: resnet18 with vgg16 retains 19.97% of isolated throughput;
lstm with resnet50 retains 19.77%; heavies retain 86-96%.

TED sensitivity recompute (measured idle, window regression): resnet18 9.32%
(r2 0.968), resnet50 9.78% (0.941), vgg16 2.32% (0.554), rnn 6.00% (0.874),
lstm 14.59% (0.993), gru 6.15% (0.905), xgboost 8.66% (0.314). These
SUPERSEDE submitted Table III (which reported 4.2-21.5% from a different,
unarchived computation path; see findings.md erratum).

TED coefficient adjustments at co-located temperature: up to +38.7%
(xgboost with resnet50, dT +19.1 C), +26.3% (rnn with vgg16, dT +19.5 C),
+16.9% (lstm with resnet50), +12.6% (resnet18 with vgg16); approximately 0
for vgg16 (calibrated at 75.6 C, already at co-located temperature).

Contamination factors (container-reported delta energy / TED-attributed
energy): up to 17.8x (lstm with resnet50), 10.4x (rnn with vgg16), 8.3x
(xgboost with resnet50), 5.7x (resnet18 with vgg16). Raw CTI (vs E_iso)
max remains 4.22 (rnn with vgg16), consistent with submitted Table V.

Negative overhead observed for lstm+gru: E_total_union 4,814 J vs summed
isolated 6,938 J. Consolidation of two similar light workloads SAVES 30.6%
energy at the cost of ~16% throughput each; conservative attribution
correctly credits the saving to both tenants.

SPP recompute matches submitted Table IV to three decimals for all seven
architectures. Table IV stands unchanged.

No TED coefficient clamps fired. 100 input files hashed. 8 artifacts
archived to data/phase6_attribution/.

### Observations

1. Adding the overhead-distribution term (toml_raw -> toml_static) INCREASES
   mean error vs the E_iso reference (31.3 -> 34.9). This is expected under
   the ground-truth semantics: E_total includes interference overhead (or
   consolidation savings), so any conservative scheme that distributes all
   of E_total must deviate from E_iso whenever E_total != sum(E_iso-scaled
   work). toml_raw is not a complete billing scheme (attributions do not sum
   to measured energy); it is the physics expectation that reveals the
   overhead explicitly.
2. TED improves the conservative scheme (34.9 -> 32.9 mean) with the gain
   concentrated where dT is large: xgboost with resnet50 58.8 -> 42.8, rnn
   with vgg16 59.8 -> 49.9, xgboost with rnn 19.1 -> 12.0. Where dT ~ 0
   (vgg16 pairs) TED is neutral to marginally negative via second-order
   shifts in the overhead split.
3. Light-container residual error under toml_ted is dominated by measured
   throughput loss, not model failure: mean toml_ted error 32.9% vs mean
   measured throughput loss 31.3%.
4. xgboost TED slope rests on a weak fit (r2 0.314; bursty inference
   pattern). Its large coefficient adjustment (+38.7%) carries this
   uncertainty and must be caveated in the paper.
5. pynvml FutureWarning on import (deprecated in favor of nvidia-ml-py);
   cosmetic, triggered via shared package import.

### Interpretation

Hypothesis partially confirmed. TED-adjusted conservative attribution
(toml_ted, 32.9% mean) beats static (34.9%) and halves the proportional
baseline error (70.2%), with max error reduced from 194.9% to 77.7%. The
improvement from TED is modest on the mean but large exactly where the
physics says it should act (large calibration-to-colocation temperature
deltas).

The decisive reframe for the paper: the ablation decomposes the gap to the
isolated reference into (a) measured throughput loss under MPS (model-free,
from batch counts; mean 31.3%), (b) thermal coefficient correction (TED,
up to +38.7%), and (c) interference overhead / consolidation savings
(signed; distributed by TO share under conservation). For conservative
TOML+TED, residual deviation from E_iso is approximately equal to the
measured slowdown, i.e. the attribution model itself contributes only
~1.6 points beyond physically real throughput loss. This directly answers
IC2E R2's 'remaining error is still high' and R1's 'metrics never enter the
model': SPP justifies TO-share distribution, TED enters the coefficients,
CTI is quantified as telemetry contamination (5.7-17.8x) plus signed
overhead.

Revision consequences: Table III regenerated from ted_recomputed.csv (old
values unreproducible from archived data; erratum logged); Table VI replaced
by the four-rung ablation; abstract/intro thermal claims ("7 to 21 percent")
replaced by the new TED range and adjustment magnitudes; conservation
(attributions sum to measured E_total) stated as a billing requirement to
motivate the overhead term.

### Artifacts

- data/phase6_attribution/ablation_results.json
- data/phase6_attribution/attribution_comparison.csv
- data/phase6_attribution/ablation_summary.csv
- data/phase6_attribution/cti_decomposition.csv
- data/phase6_attribution/host_crosscheck.csv
- data/phase6_attribution/ted_recomputed.csv
- data/phase6_attribution/spp_recomputed.csv
- data/phase6_attribution/run_metadata.json
