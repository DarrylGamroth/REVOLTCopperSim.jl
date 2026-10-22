# Paired atmospheric discretization results

Finite 960-pixel master pupil, native 3n periodic atmosphere screens; all grids sample the same screens. The 3.66 m canvas omits frequencies below 0.273 m⁻¹, above 1/L0 = 0.033 m⁻¹; no subharmonics restore this low-frequency power. Winds travel up to 6 m in 0.6 s, reusing periodic structure. The 1440 / 256 comparison is a numerical reference, not an independent physical oracle. No detector noise or pixel quantization is applied to the host controller. Primary cases use their own calibration with a 5% singular-value cutoff and fixed gain 0.15. Optional anchors may reuse the 1440 / 256 calibration, as recorded. Science diagnostics use a common 480 grid with sparse 960 and 1440 checks. This exercises a causal host controller and plant, with no actual pyRTC integration. Timing covers only the warmed static plant endpoint; controller, atmosphere generation, transfers and science diagnostics are excluded.

Reference: 1440 / 256. Burn-in: 50 frames; frame k uses the command applied before the next controller update. Sampling is paired by seed, r0, screen hashes and source/controller metadata. Open slopes have each case's flat response removed.

Open exposures are sampled at frame 1, every tenth frame and the last frame (31 samples for a 300-frame run); closed trajectories and science diagnostics retain every update. Open error statistics describe only those sampled frames. Post-burn open metrics select actual frame IDs greater than the burn-in (25 samples at burn 50 for 300 updates). Optional anchor runs are plant sensitivity checks with the reference calibration, without concurrent fine-system same-command exposures. Relative errors are aggregate Frobenius norms and temporal median, p95 and worst frame L2 errors. Absolute slope RMS is dimensionless; rates retain the archived photon-rate units. Total photon rates and their paired throughput errors are also reported separately from image-shape errors. Undefined relative errors (zero reference signal with nonzero error) are null and counted. Closed trajectory differences include the accumulated controller response; sparse same-command checks isolate the plant response to identical applied commands.

Each seed is one replicate within each r0. Temporal summaries are descriptive; frames are correlated and are not independent replicates. Reusing a seed across r0 also does not create independent atmosphere realizations. Seed dispersion uses sample standard deviation; a single seed has no dispersion estimate.

Coverage: 47 runs, 9 seed/r0 pairs. Partial coverage allowed: False. See metrics.json for exact coverage and calibration provenance.

| Grid / points | r0 (m) | Seeds | Open rate error | Open slope error | Residual RMS (nm), seed mean ± SD | Mean Strehl | Residual RMS / reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 180 / 64 | 0.10 | 3 | 1.306% | 1.717% | 120.3 ± 11.204 | 0.16251 | 0.9908 |
| 180 / 64 | 0.15 | 3 | 1.450% | 1.763% | 84.497 ± 7.9376 | 0.39455 | 0.99411 |
| 180 / 64 | 0.20 | 3 | 1.573% | 1.811% | 66.278 ± 6.2668 | 0.56085 | 0.99602 |
| 240 / 96 | 0.10 | 3 | 0.614% | 0.636% | 121.33 ± 11.541 | 0.15857 | 0.99915 |
| 240 / 96 | 0.15 | 3 | 0.608% | 0.674% | 84.968 ± 8.1038 | 0.3908 | 0.99957 |
| 240 / 96 | 0.20 | 3 | 0.605% | 0.715% | 66.531 ± 6.3594 | 0.55844 | 0.99975 |
| 240 / 128 | 0.10 | 3 | 0.608% | 0.653% | 121.61 ± 11.625 | 0.15758 | 1.0014 |
| 240 / 128 | 0.15 | 3 | 0.603% | 0.659% | 85.09 ± 8.138 | 0.38986 | 1.001 |
| 240 / 128 | 0.20 | 3 | 0.606% | 0.672% | 66.595 ± 6.3748 | 0.55785 | 1.0007 |
| 480 / 32 | 0.10 | 3 | 1.356% | 2.122% | 120.75 ± 11.417 | 0.16105 | 0.99438 |
| 480 / 32 | 0.15 | 3 | 1.637% | 2.566% | 84.724 ± 8.0615 | 0.3929 | 0.99671 |
| 480 / 32 | 0.20 | 3 | 1.787% | 2.941% | 66.405 ± 6.3367 | 0.55971 | 0.99787 |
| 1440 / 256 | 0.10 | 3 | 0.000% | 0.000% | 121.43 ± 11.559 | 0.15818 | 1 |
| 1440 / 256 | 0.15 | 3 | 0.000% | 0.000% | 85.005 ± 8.1066 | 0.39052 | 1 |
| 1440 / 256 | 0.20 | 3 | 0.000% | 0.000% | 66.547 ± 6.3584 | 0.55829 | 1 |
| 1440 / 512 | 0.15 | 1 | 0.025% | 0.089% | 94.1 (one seed) | 0.31269 | 1.0003 |
| 1920 / 256 | 0.15 | 1 | 0.039% | 0.028% | 94.067 (one seed) | 0.31297 | 0.9999 |

Science grid errors hold each run's command fixed. Their sparse samples at frame 1, every 50 frames and the last frame do not bound unsampled frames. metrics.json reports 480 versus 960/1440 RMS errors and Strehl differences, plus 960 versus 1440 differences. summary.csv contains one row per seed/r0/case. No automatic accuracy threshold or production preset recommendation is inferred from this finite ensemble.
