# Copper atmospheric convergence: independent numerical review

Review scope: paired finite-screen atmospheric input, spatial/modulation convergence, closed-loop trajectories, and common-grid science metrics. No GPU workloads and no production modifications performed. Review based on Copper revision a82af57aaeab99b2d58de6cb87edacf9c6af7414 and the current local AOS source. Experiment script review pending.

## Accepted design, derived from source

- AOS `src/atmosphere/multilayer.jl:303` centers each extracted grid at `(m+1)/2`, with increment `footprint_scale`. With master pupil size N=960 and candidate size n, factor N/n produces coordinates `(m+1)/2 + (N/n)*(i-(n+1)/2) - offset`. Those are physically coincident cell-center grids across the same aperture, not endpoint-aligned interpolations. Independent FFT screens at each n would not provide this paired comparison.
- AOS `src/atmosphere/kolmogorov.jl` stores phase in radians at the reference wavelength. Each layer is generated with the total r0, then multiplied by sqrt(Cn2 fraction). The proposed OPD conversion sqrt(weight) × 500 nm/(2π) is correct. Do not additionally rescale r0 per layer or double-apply Cn2 weights. The optical propagation wavelength remains 550 nm.
- Offsets in master pixels are wind velocity × time / (1.22/960 m). Compute times from the frame index, so the paired trajectories do not accumulate grid-dependent clock errors. AOS subtracts offsets from sampling positions; preserve that sign consistently.
- The composition adds DM OPD to atmospheric OPD. An interaction matrix calibrated with positive DM pokes therefore requires c[k+1] = c[k] − gain × R × (s[k] − s_flat). The command unit is OPD metres, with no extra reflection factor.
- With singular values truncated at 5% of each matrix's largest singular value, each plant has its own valid simulated reconstructor. This tests calibration-plus-plant performance. The retained rank can differ between candidates and must be reported.
- Common science-grid evaluation should sample the same atmospheric field and evaluate the same continuous Gaussian DM functions directly, using the candidate command at that frame. Bilinearly resizing candidate residual OPD would suppress unresolved residual structure and bias the comparison.

## Findings and acceptance conditions

### ATM-01 — finite canvas substantially limits physical seeing interpretation

Severity: limitation; confidence: high; evidence: observed source and derived scales. AOS uses 3N periodic screens without subharmonics in this backend. At N=960 the screen is 3.66 m wide, fundamental frequency 0.273 m⁻¹, while L0⁻¹ is 0.0333 m⁻¹. Missing low spatial frequencies alter low-order wavefront power; nominal r0 does not repair this. At 10 m/s a 0.6 s trace translates 6 m, exceeding one screen width (not necessarily an exact 2D recurrence for oblique winds).

Disposition: acceptable for numerical convergence within this stated model, not validation of full atmospheric statistics or a physical seeing envelope. Required reporting: actual open-loop piston-removed pupil RMS, fixed master-screen resolution and interpolation, missing low-frequency power, periodic reuse, and absence of detector noise. Do not call the fine model an oracle.

### ATM-02 — trajectories and same-command discrepancies answer different questions

Severity: methodological acceptance condition; confidence: high. Independently controlled plants correctly measure each discretization's closed-loop behavior. Differences in their photon-rate or slope trajectories include different commands, so they cannot isolate instantaneous sensor discretization error.

Disposition: retain independent closed loops for residual and command comparisons. Use identical atmosphere and identical command for open-loop/snapshot photon-rate and slope comparisons. Inject selected candidate commands into the fine plant at matching timestamps as planned. Record which command owner generated every snapshot. Compare candidate-owned science residuals against fine-owned science residuals separately.

### ATM-03 — command/time alignment must remain causal

Severity: correctness acceptance condition; confidence: high. At frame k, render atmosphere(t[k]) + DM(c[k]), measure s[k], record residual science metrics and c[k], then form c[k+1]. A saved post-update command paired with the preceding residual introduces a one-frame mismatch.

Required validation: zero initial command; at least one controller update checked algebraically; first science residual equals open-loop atmosphere; logs/snapshots identify whether stored commands are applied or next commands. Burning 50 frames means excluding frames 1–50 from every corresponding summary.

### ATM-04 — science metric conventions must be explicit

Severity: reporting acceptance condition; confidence: high. Compute pupil RMS after piston removal with the same fine-grid mask for all candidates. Exact on-axis normalized coherent intensity is |mean(exp(i 2π OPD/λ))|² for a uniform binary pupil. This is an on-axis Strehl metric; it includes image-motion loss and is different from a recentered PSF peak. The approximation exp(−variance(phase)) can fail for poor correction and should not replace the coherent metric.

Required reporting: science wavelength, common mask, piston treatment, on-axis/recentered convention. Do not subtract tip/tilt unless declared as a distinct metric.

### ATM-05 — paired samples do not provide hundreds of independent experiments

Severity: uncertainty acceptance condition; confidence: high. Adjacent frames share frozen-flow screens and commands; seeing amplitudes generated with the same random seed are paired/scaled inputs. Three seeds per seeing are three realizations, not 750 independent realizations after burn-in.

Required reporting: per-seed summaries and spread; paired differences across discretizations. Avoid frame-wise confidence intervals that assume independence or broad probability claims from three seeds. A 300-frame record is a 0.6 s finite-duration test.

### ATM-06 — fine-plant numerical checks need matching input and duration

Severity: reference acceptance condition; confidence: high. Spatial refinement 1440→1920 and modulation refinement 256→512 isolate distinct reference sensitivities. Both should consume matched atmosphere and command inputs for instantaneous tests. If run as independent controllers, label their differences as complete-loop differences, and summarize duration/seed coverage explicitly.

Disposition: a sparse snapshot check can support a bounded reference claim for those samples; it cannot establish full-trajectory convergence by itself.

## Script review

Pending creation of atmospheric experiment script. Inspect calibration provenance (including common divisor), complete 277 columns, command and slope shapes/order, screen generation/weights, time alignment, rank cutoff, metric normalization, applied-command snapshots, and caching metadata before production runs.

## Script review, first implementation

Reviewed `/home/dgamroth/workspaces/codex/REVOLTCopperSim-atmospheric/benchmarks/copper_atmospheric.jl` (153-line initial version).

Observed correct in source:

- `science` adds DM surface to atmospheric OPD (line 28), removes piston for RMS using population standard deviation, and uses coherent on-axis intensity at 550 nm.
- Gaussian basis and command scatter match AOS `build_separable_influence!`: normalized pixel centers `(2i−n−1)/n`, sigma = Copper influence width, pitch 0.125, centered 19×19 grid, exact actuator-grid indices. The science aperture matches the binary circular telescope aperture in exact arithmetic. Float32 versus Float64 boundary and basis rounding remain a validation item.
- Initialization generates the five independent layers at r0=0.15 m, then `render!` scales phase by `(0.15/r0)^(5/6)`. This follows the square root of the native PSD's r0^(−5/3) factor, and intentionally pairs realizations across nominal r0 values.
- Times are frame×0.002 s (first recorded time 0.002 s). Commands are saved before the causal update, and science diagnostics use that applied command.
- Flat signals are taken with zero command and zero atmosphere for each sensor configuration, while all slopes use the common reference divisor. Interleaving zero-command open-loop exposures with closed-loop exposures is valid because the calibration graph has static OPD input, noiseless acquisition, and no atmospheric or controller state advancement.

### ATM-07 — validate reused calibration metadata before inversion

Severity: medium; confidence: high; classification: observed missing integrity/provenance guard, not evidence of incorrect existing calibration. `calibration()` currently checks only matrix shape. A stale full matrix or matrix with a different slope divisor has the same dimensions and will silently change loop gain/response. Validate `full_calibration`, exact column ordering 1:277, archived array SHA256, common divisor, and source graph/AOS compatibility. If archived revisions differ only through harmless metadata changes, adjudicate and record that explicitly rather than suppressing the mismatch. Persist calibration-file and matrix hashes with each atmospheric run.

### ATM-08 — common science grid changed from 1440 to 480

Severity: medium acceptance condition; confidence: high; classification: observed design change. Common grid 480 removes candidate-specific science quadrature differences, but does not establish quadrature convergence of absolute residual or Strehl. Planned 960 checks at frame 1 and final frame are useful. Add 1440 checks at those samples: the underlying bilinear continuous screen can retain quadrature effects even at the master cell-center resolution 960. Record per-snapshot absolute RMS and Strehl differences, and interpret any candidate-versus-reference differences alongside this error. Persist science-grid resolution and wavelength in run metadata.

Required targeted checks before the main run:

1. CPU versus GPU extraction of the same nontrivial screen on all candidate grids, including a nonintegral shift crossing the periodic boundary. A small analytic screen can establish interpolation/sign/centering independently; a seeded native screen exercises the production data path.
2. Exact DM fixture versus actual graph surface for asymmetric nonzero physical commands, with the comparison restricted to the same pupil and attention to the grid transpose/order. A symmetric or single central poke would not detect axis reversal.
3. First closed-loop frame matches its open-loop signal and science residual with zero command; next applied command equals the specified update from the first closed-loop signal. This catches reset and command-timestamp mistakes.
4. Common science 480/960/1440 sample comparison, including a nonzero settled command. Record the numerical error, not merely a passing tolerance.
5. Fine same-command sensor snapshots explicitly reset the reference flat state and render atmosphere at the matching timestamp; distinguish them from the independent reference controller trajectory.

No confirmed defect in the implemented phase units, layer weights, causal update, sign, or DM basis was found. The above metadata guards and numerical checks remain prerequisites for a strong result claim. No GPU jobs were run by the reviewer.

## Final source pass before full experiment

Reviewed the 201-line final Julia harness and `test/benchmarks/atmospheric_parity.jl`. No numerical blocker was found in this source pass.

- ATM-07 is addressed for newly computed trajectories: full 277-column ordering, common divisor, graph hash/AOS revision, and compressed interaction-array checksum are checked before inversion. Existing cache reuse checks harness/graph/AOS identity, screen hashes, frame count, and gain. Cached calibration-file replacement is outside those checks; preserve the supplied calibration archive immutable.
- ATM-08 now has paired 480/960/1440 science evaluation at frame 1, every 50th frame, and the last frame. Actual differences still need assessment against reported candidate effects.
- Open-loop WFS exposures are reduced to frame IDs `[1; 10:10:300]`, while closed-loop feedback remains at every 2 ms frame. This does not change the controller cadence. Post-burn open-loop comparisons must select physical frame IDs greater than 50, yielding 25 open samples per full trajectory; selecting columns greater than 50 would be wrong.
- Commands are stored before update. The first closed exposure is asserted equal to the first open exposure, and a whole-record assertion checks recorded command increments against recorded closed-loop signals.
- Candidate same-command exposures use the fine 1440/256 sensor with the identical command and atmospheric timestamp. Both own and fine raw rates are retained at these samples.
- Optional 1920/256 and 1440/512 anchors explicitly reuse the 1440/256 reconstructor and record that calibration source. Their results assess plant sensitivity under a fixed reference controller; they must not be described as independently recalibrated anchor plants.
- Source inspection confirms the targeted tests exercise nonzero seeded fields, r0 amplitude scaling, CPU/CUDA parity at an ordinary shift and a large wrapping shift, asymmetric DM commands, and plus-DM graph composition. The primary reports 9/9 checks passed; this reviewer did not rerun GPU work.

Remaining work is result interpretation: quantify science quadrature errors; compare references/anchors on matched samples; report actual open-loop RMS, finite-screen and periodicity limitations, ranks, per-seed spread, and finite-duration/noiseless-controller scope. This review does not certify physical atmosphere accuracy or operational RTC calibration.

Final reviewed Copper commit: `3dcbd48d0042325ffdeaea0996781ff72942ffe8`.
Final reviewed atmospheric harness SHA256: `f0c9f2e384cc6eee5e7e8c487e137eb441e17bfa29b19bacc6999d76c1215669`.


## Independent offline evidence review: seed 1

Scope: all 15 completed seed-1 records (five cases × three nominal r0 values), read directly from `main/`. No GPU work or production/reducer edits were performed. Seeds 2 and 3 and optional anchors were outside this data review. The independent script and detailed numerical results are retained as `independent-review-seed1.py` and `independent-review-seed1.json` in this evidence directory.

Reviewed reducer SHA256: `0ab4c7a3b388f5fd1946557f5cbbd083dc15a8b107aed53bf5e4dba255f9de1f`.

Observed checks:

- Verified compressed-array SHA256 before decoding Julia column-major Float32 arrays. Reconstructed slopes with explicit independent pixel loops, four quadrant formulas, column-major support ordering, and each sensor's flat subtraction. The largest difference from archived Float32 slopes was 5.96 × 10⁻⁸, consistent with output rounding.
- Recomputed each of the five own-case truncated SVD inverses from the previous full calibration. Every case retains 231 of 277 singular directions. Recomputed all 299 applied-command transitions per trajectory; maximum absolute discrepancy was 2.99 × 10⁻¹⁴ m. Every initial command is zero and first-frame residual science diagnostics equal their open-loop counterparts.
- Confirmed 300 closed-loop updates, 31 sampled open exposures, and 25 open exposures after physical frame 50. Every paired case has exactly identical archived common open-loop science diagnostics and screen hashes.
- Independently recomputed all 84 candidate same-command snapshots (four candidates × three r0 × seven checkpoints), including rate L2 errors, flat-subtracted slope errors and absolute slope RMS. They agree with harness metadata and the reducer. Archived own snapshot rates reproduce the corresponding recorded closed slopes.
- Compared independent aggregate open rate/slope errors, post-burn sampled open errors, residual mean ratios and mean Strehl against the reducer for every record; all agreed to the specified tight numerical tolerances. The reducer handles physical open-frame IDs and correlated-seed summaries correctly in inspected paths.
- All sampled rates are nonnegative. Maximum total open photon-rate departure from each case's flat total is 2.39 × 10⁻⁷ fraction.

### Observed first-seed results

Rate/slope errors below are aggregate relative L2 errors over all 31 open samples. Residual and Strehl means use frames 51–300. Residual ratio divides each candidate's mean RMS by the reference mean RMS at the same nominal r0. The same-command slope column is the worst among seven sparse matched-command checkpoints, including initialization.

| Grid / points | r0 (m) | Open rate error | Open slope error | Residual RMS (nm) | Residual ratio | Mean on-axis Strehl | Worst same-command slope error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 180 / 64 | 0.10 | 1.266% | 1.734% | 132.844 | 0.988766 | 0.108562 | 2.076% |
| 240 / 96 | 0.10 | 0.616% | 0.634% | 134.235 | 0.999115 | 0.105347 | 0.903% |
| 240 / 128 | 0.10 | 0.609% | 0.638% | 134.597 | 1.001812 | 0.104683 | 0.921% |
| 480 / 32 | 0.10 | 1.284% | 2.044% | 133.488 | 0.993556 | 0.107508 | 3.289% |
| 1440 / 256 | 0.10 | 0.000% | 0.000% | 134.354 | 1.000000 | 0.105136 | reference |
| 180 / 64 | 0.15 | 1.403% | 1.811% | 93.391 | 0.992722 | 0.317522 | 2.005% |
| 240 / 96 | 0.15 | 0.611% | 0.682% | 94.039 | 0.999612 | 0.313091 | 0.894% |
| 240 / 128 | 0.15 | 0.605% | 0.646% | 94.196 | 1.001274 | 0.312151 | 0.892% |
| 480 / 32 | 0.15 | 1.573% | 2.490% | 93.727 | 0.996287 | 0.315499 | 4.032% |
| 1440 / 256 | 0.15 | 0.000% | 0.000% | 94.076 | 1.000000 | 0.312903 | reference |
| 180 / 64 | 0.20 | 1.525% | 1.883% | 73.298 | 0.995073 | 0.490078 | 1.972% |
| 240 / 96 | 0.20 | 0.607% | 0.733% | 73.648 | 0.999827 | 0.486889 | 0.886% |
| 240 / 128 | 0.20 | 0.607% | 0.662% | 73.726 | 1.000885 | 0.486233 | 0.874% |
| 480 / 32 | 0.20 | 1.719% | 2.881% | 73.480 | 0.997546 | 0.488503 | 4.554% |
| 1440 / 256 | 0.20 | 0.000% | 0.000% | 73.661 | 1.000000 | 0.486799 | reference |

Common open-loop mean pupil RMS is 452.205, 322.547, and 253.791 nm for nominal r0 = 0.10, 0.15, and 0.20 m, respectively. These are measured values for the finite native screen model, not validation of the missing low-frequency atmospheric power.

### Science-grid quadrature

Across all 105 recorded science checkpoints, largest absolute 480→1440 difference is 0.01565 nm RMS and 9.67 × 10⁻⁵ on-axis Strehl. Largest 960→1440 difference is 0.01262 nm RMS and 8.01 × 10⁻⁵ Strehl. These are observed sparse-sample discrepancies, not bounds on every frame or proof of exact continuous-screen quadrature.

For comparisons of candidate minus reference, common-grid errors cancel strongly. Recomputing the paired effect at each checkpoint on both grids changes that effect by at most 0.001429 nm RMS and 6.22 × 10⁻⁶ Strehl across all candidates (largest values occur for 480/32). For 240/96 alone the corresponding maxima are 3.16 × 10⁻⁵ nm and 3.26 × 10⁻⁷. Thus the measured checkpoint comparisons are not explained by the common science-grid quadrature discrepancy. This does not establish convergence of unsampled frames.

### Interpretation and remaining acceptance conditions

The 180/64 candidate has a slightly lower residual than the reference while its sensor slopes differ by approximately 1.7–1.9% on these open samples. This does not make it the more faithful optical model. Its independently calibrated plant and controller produce a different feedback response; a lower RMS or higher Strehl under a fixed gain can occur without more accurate sensor propagation. All retained ranks are equal here, so different retained rank counts do not explain this observation. No causal mechanism beyond the observed plant/controller difference has been established; do not promote optical-gain or aliasing explanations to confirmed causes without a discriminating experiment.

The 240/96 and 240/128 plants have smaller measured sensor discrepancies than 180/64 or 480/32 for this first seed. Their mean residuals are also close to the reference. This is evidence within the tested finite-screen ensemble, not a universal preferred preset or a physical-accuracy claim. The 1920/256 and 1440/512 anchors have not yet been reviewed and remain necessary for any claim that 1440/256 is sufficiently converged for these quantities.

Reducer reporting follow-ups communicated to the primary:

1. Explicitly state that this native atmosphere omits subharmonics and therefore omits low-frequency power below the 3.66 m canvas fundamental. Merely saying 'finite periodic screen' does not communicate the significance of the limitation.
2. Verify the intended full ensemble independently before declaring completion: seeds {1,2,3} × nominal r0 {0.10,0.15,0.20} × five primary cases. The reducer's current nonpartial mode requires all five cases for each discovered seed/r0 pair but does not require all nine planned pairs. This is a coverage guard limitation, not a wrong calculation in the 15 reviewed records.
3. Keep optional fixed-reference-calibration anchor results distinct from independently calibrated primary cases, and report sparse same-command, open-loop, and independent closed-loop quantities separately.

The identified 3.955 GiB dense pupil-phase allocation is a separate observed implementation-memory issue. The source trace and failed preparation allocation support an allocation-cost finding; they do not establish a leak, a speedup, or any change in numerical fidelity. Process isolation preserves the frozen numerical harness and does not count as remediation of AOS ownership.


## Independent full primary-ensemble verification

Completed offline verification of all 45 primary records: seeds {1,2,3} × nominal r0 {0.10,0.15,0.20} × cases {1440/256, 480/32, 180/64, 240/96, 240/128}. This includes the previously reviewed first seed and independently checks seeds 2 and 3. Reproducible verifier: `independent-review-main.py`; detailed recomputed values: `independent-review-main.json`. No Julia, GPU, or production workloads were launched by this reviewer.

All records share the frozen harness SHA256 `f0c9f2e384cc6eee5e7e8c487e137eb441e17bfa29b19bacc6999d76c1215669` and matching base harness, graph, AOS/Copper revision, cadence, controller and science metadata. Each seed has one identical five-screen hash tuple across all cases and strengths; each tuple contains five different screen hashes, and the three seeds have different tuples. These are content-identity checks; seed independence follows from the explicit initialization protocol rather than from hashes alone.

All array checksums, slope reconstructions, rank checks, causal transitions and saved-reducer comparisons passed. Every inverse retains 231 directions. Across 13,455 recorded causal transitions, the largest independent update discrepancy is 3.05 × 10⁻¹⁴ m. Maximum archived open-slope rounding discrepancy is 5.96 × 10⁻⁸. Maximum total open-rate deviation from its own flat total is 2.58 × 10⁻⁷ fraction. All 252 candidate matched-command snapshots were recomputed successfully. Main-analysis contains exactly 45 runs and nine seed/strength pairs, and its principal per-run metrics agree with the independent reduction.

### Full-ensemble sensor and closed-loop comparisons

Ranges span the nine paired seed/strength run aggregates. Open errors use all 31 sampled frames; residual ratios use means over frames 51–300. Sparse worst slope errors use matched atmospheric timestamps and identical applied commands in candidate and reference sensors.

| Case | Open rate error range | Open slope error range | Residual RMS ratio range | Worst same-command slope error |
| --- | ---: | ---: | ---: | ---: |
| 480 / 32 | 1.284–1.856% | 2.044–3.009% | 0.993556–0.998762 | 4.554% |
| 180 / 64 | 1.266–1.612% | 1.628–1.883% | 0.988766–0.997236 | 2.196% |
| 240 / 96 | 0.599–0.626% | 0.596–0.753% | 0.998999–0.999836 | 0.952% |
| 240 / 128 | 0.591–0.620% | 0.638–0.688% | 1.000522–1.001812 | 0.921% |

After restricting open samples to physical frames greater than 50, open slope error ranges are 2.018–3.033% (480/32), 1.664–1.862% (180/64), 0.616–0.757% (240/96), and 0.630–0.683% (240/128). These are descriptive errors over 25 sampled frames per run, not all 250 post-burn frames.

Reference mean residual RMS across the three seeds is 121.433 ± 11.559 nm, 85.005 ± 8.107 nm, and 66.547 ± 6.358 nm for nominal r0 0.10, 0.15, and 0.20 m, respectively (mean ± sample standard deviation across seeds). Open-loop mean RMS varies substantially by realization: 306.374–452.205 nm, 218.529–322.547 nm, and 171.947–253.791 nm for those strengths. This demonstrates why paired differences and seed dispersion are useful and why nominal r0 alone does not describe the realized finite screen.

### Full-ensemble science-grid sensitivity

Across 315 science checkpoints, maximum absolute 480→1440 discrepancy remains 0.01565 nm RMS and 9.67 × 10⁻⁵ on-axis Strehl. Maximum 960→1440 discrepancy is 0.01679 nm RMS and 8.82 × 10⁻⁵ Strehl. The intermediate grid is not uniformly closer on every realization; these sampled quadrature discrepancies should not be described as a monotone convergence proof.

Maximum changes in the paired candidate-minus-reference effect when measured on 1440 instead of 480 are:

| Candidate | RMS effect change (nm) | On-axis Strehl effect change |
| --- | ---: | ---: |
| 480 / 32 | 0.001429 | 6.22 × 10⁻⁶ |
| 180 / 64 | 0.000329 | 3.36 × 10⁻⁶ |
| 240 / 96 | 0.0000640 | 6.40 × 10⁻⁷ |
| 240 / 128 | 0.0000720 | 8.42 × 10⁻⁷ |

These are maxima over observed sparse checkpoints. Common-grid errors largely cancel in the paired comparisons, but no claim is made for unobserved times or an exact continuous-field integral.

### Reporting follow-up disposition

The current reducer explicitly documents missing low-frequency atmospheric power and lack of subharmonics. The current persister requires 47 final records, exactly the nine planned seed/r0 pairs, five primary cases per pair, and exactly two anchors at seed 1/r0 0.15. Its source also checks that every archived run matches the frozen harness. These changes address the earlier reporting and final-coverage guard requests. Persister acceptance remains pending actual anchor completion and final archive execution.

Reviewed current reducer SHA256: `6ea40790264c2177469cca5a4aa885171f95bbeda479611b620b64188cb2f5fd`.
Reviewed persister SHA256: `34ed14a6cc4f77b54e8f41b1446e10eead3ae87667d404df8b39e945f6930e21`.

No numerical blocker was identified in the completed primary ensemble. The earlier interpretation remains: lower residual RMS in a different independently calibrated loop is not evidence of greater optical fidelity. Both fine anchors still require separate verification before any reference-refinement conclusion.

## Final anchor and archive verification

Final numerical disposition: no unresolved numerical blocker was found in the 47-record atmospheric campaign. The complete primary ensemble and both optional anchors have now received independent offline verification. This is software/data verification of the stated finite-screen model and simulated host controller; it does not validate physical atmospheric statistics, measured mirror calibration, operational pyRTC behavior, noisy control, or real-time deadlines.

Verified archive: `/home/dgamroth/workspaces/codex/REVOLTCopperSim-atmospheric/benchmarks/results/2026-10-21-copper-atmospheric`. Reproducible final verifier: `independent-review-final.py`; machine-readable findings: `independent-review-final.json` in the evidence directory.

### Archive integrity and coverage

- All 833 raw files match the archive's SHA256 manifest. All 30 supporting record files present in the reviewed archive snapshot also match their manifest, with exact file-set equality in both directories. The primary will repersist this final review and verifier afterward, so the supporting-record count will increase; the raw numerical dataset remains the same.
- Exact 47-record coverage is present: all 45 primary Cartesian combinations, plus 1920/256 and 1440/512 only at seed 1 and nominal r0 0.15 m. There are exactly nine seed/strength pairs, and metadata/run-file keys agree with the saved reducer.
- All four recorded source hashes match the current harness, driver, reducer and persister. The numerical Julia harness remains SHA256 `f0c9f2e384cc6eee5e7e8c487e137eb441e17bfa29b19bacc6999d76c1215669`. The analyzer hash recorded in metrics agrees with its source hash. Every run's metadata-file checksum agrees with the saved metric record.
- Both anchor records match the reference's seed, five-screen hashes, source identities, time sampling, controller coefficients, normalization and science settings. Both explicitly use the 1440/256 calibration with 231 retained directions. Their labels correctly identify fixed-reference-calibration plant sensitivity; same-command sensor snapshots are explicitly unavailable for these two isolated anchor runs.
- Recomputed integrated photon totals for all 47 records and verified the new throughput metrics separately from image Frobenius errors. Every archived open photon-rate frame and flat frame is nonnegative. Maximum paired total-photon-rate discrepancy is 6.68906 × 10⁻⁵ fraction, or 0.00668906%, agreeing with README rounding to 0.00669%.
- The previously verified 45 primary numerical results are unchanged in the final saved reductions.

### Independently recomputed anchors

All percentages below compare against 1440/256 at seed 1, nominal r0 0.15 m. Open quantities use 31 sampled frames; closed-loop science differences use frames 51–300. The commands evolve separately in each plant, using the same reference reconstructor.

| Anchor | Open image error | Open slope error | Closed-trajectory slope error | Command-trajectory error | Mean residual RMS difference | Mean on-axis Strehl difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1920 / 256 | 0.038969% | 0.028484% | 0.035009% | 0.014041% | −0.0092563 nm | +0.00006916 |
| 1440 / 512 | 0.024668% | 0.089033% | 0.058147% | 0.061145% | +0.0241642 nm | −0.00021181 |

All compared open, post-burn open, closed-trajectory, command-trajectory, mean residual and mean Strehl fields agree with independently recomputed saved reductions. Shared-controller causal-update discrepancies are at most 2.61 × 10⁻¹⁴ m and 1.97 × 10⁻¹⁴ m for the two anchors. Flat-subtracted open slopes were independently reconstructed from raw pixel rates and checked against archived slopes.

Anchor absolute 480→1440 science RMS discrepancies are about 0.01126 nm, but the paired anchor-minus-reference effect changes by at most 7.14 × 10⁻⁶ nm and 1.04 × 10⁻⁵ nm when evaluated at 1440 instead of 480. The common-grid quadrature discrepancy therefore largely cancels in these matched science comparisons at the observed checkpoints. This supports interpreting the reported anchor effects; it does not turn sparse checkpoint errors into full-trajectory bounds.

### Scope of the refinement conclusion and README review

For the one tested anchor realization/strength, further spatial or modulation refinement changes the measured sensor and loop quantities much less than the primary candidate-versus-reference differences. This supports using 1440/256 as a numerical comparison for that anchor case. The anchors cover one seed and one strength, and separately refine one axis; they do not establish a universal error bound, simultaneous two-axis convergence, independently recalibrated fine-loop performance, or an independent physical oracle.

The final README accurately separates image error from total throughput, loop performance from numerical fidelity, and plant service time from full RTC latency. Its table matches the independent primary-ensemble results at the stated rounding. It reports the finite-screen/subharmonic limitation, correlated samples, own versus shared calibration, unchanged defaults, prior spatial-frequency limitations, process-isolation recovery, and explicit timing exclusions. No unconditional production replacement or physical-validation claim is made.

Final verified source hashes:

- Analyzer: `668a48d4dd8a4b0cd19e37872c75d3ee374503aaeb473932475e8c435a3320bd`.
- Persister: `cbd4d1effcc82e27816f368ef71c9efce30854449b9807448ab3d719c17ae086`.
- Driver: `3228ffae3da75929e1d5141919af6fdc4c26a349fc169e4e1fa8e013d47f4df8`.

No production changes or GPU work were performed by the numerical reviewer. Ready for the primary to repersist this final review and verifier artifacts and confirm the refreshed supporting-record manifest.
