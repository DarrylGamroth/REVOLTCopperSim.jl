# Independent source and numerical review

2026-10-21. Read-only Astra review by agent `/root/copper_convergence_review`.
No CUDA work, source edits or production tuning were performed by the reviewer.

## Confirmed and resolved findings

- R1 — AOS arrays use `(x,y)` while FGA images use `(row=y,column=x)`.
  The main/frequency reducers explicitly transpose their indexing. Independently
  checked native-to-FGA calibration permutation is exact; photon arrays unchanged.
- R2 — Original early persistence could accept incomplete timing records. The
  frozen full harness now writes metadata after all stages and checks its own
  hash, `graphs.jl` hash, AOS HEAD and pilot/full/benchmark flags on reuse.
- R3 — Only 45 of 48 full cases have timing/allocation records; the three finest
  reference/anchor cases are numerical only. Report wording corrected.
- R4 — High-frequency reference is not converged: 1920/256 changes the 192/191
  response by 14.07% (absolute RMS 7.074×10⁻⁵), and 96/95 by 10.97%. The 512-point
  angular check changes 192/191 by 0.811%. Report keeps this uncertainty explicit.
- R5 — Archived historical pilot is optional for reproduction. Cache guards are
  described precisely; changed dependency/environment state requires a new dir.

## Source-level checks

Physical sample coordinates, normalized pupil units, fixed detector geometry,
common reference incidence normalization, FGA I 4 Q channel order, per-actuator
command clearing, finite central differences and completed CUDA timing endpoints
are consistent with the experiment. Public defaults remain unchanged.

The 180/64 improvement applies to the tested low-frequency probes (up to 12 cycles
per aperture) and smooth Gaussian DM responses. It is not a guarantee for all
inputs below Nyquist. Even 32/31 and 64/63-cycle probes below 180-grid Nyquist
have 45.6% and 119.1% relative response errors. The 192/191 pattern is representable
on 480 and aliases exactly to 12/11 on 180 (independently checked to 7.35×10⁻¹⁴).
Its 180/64 absolute response error 0.03755 is far above reference response
RMS 0.0005028, explaining 7469% relative error. The 480/32 response error is also
101.2%; representability is necessary but insufficient for propagation accuracy.

Focal sample spacing remains Npupil/Nfft=30/64=0.46875 λ/D. The sweep does not
refine focal spacing or qualify the shifted-mask approximation. Multiscale 100 nm
is a coefficient, not RMS, and its spectrum is not an atmospheric ensemble.

## Independent data verification

- All 48 full photon-rate/interaction array pairs checksummed. Four principal
  metrics independently recomputed: rates, flat-subtracted slopes, matrix and
  worst-column errors. Maximum discrepancy from reported values 7.27×10⁻¹⁵.
- All 9 spatial-frequency arrays checksummed/source-matched; all 54 response RMS
  values independently verified.
- All 18 repeat timing records checked: 72 endpoints,9216 timing samples and 1152
  zero-host-allocation observations. No timed GC. HIL medians 480/32=2.6540395 ms,
  180/64=0.802908 ms. Source fingerprints match the frozen main harness and model.
- No remaining numerical blocker for the qualified conclusions. Production
  default change, atmospheric statistics and closed-loop qualification deferred.
