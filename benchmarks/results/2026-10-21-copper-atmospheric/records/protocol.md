# Paired atmospheric and AOS implementation investigation

2026-10-21. Production sampling defaults and algorithms are unchanged.

## Scientific comparison

- Frozen harness commit: `3dcbd48` in REVOLTCopperSim.jl.
- Native AOS five-layer von Kármán finite-screen generator, master pupil 960,
  2880² screens, physical canvas 3.66 m, explicit SplitMix64 seeds 1, 2, 3.
- Native Copper weights, layer winds and L0=30 m; r0=0.10, 0.15, 0.20 m at
  500 nm. Strengths scale the same per-seed realization by r0⁻⁵⁄⁶.
- All grids sample the same continuous piecewise-bilinear frozen-flow screens,
  with exact centered physical coordinates and matched 2 ms times.
- 300 causal feedback updates per case/seed/strength; burn-in 50. Open-loop
  rates/slopes sampled at frame 1 and every 10 frames (31 exposures per run).
- 480/32, 180/64, 240/96, 240/128 compared with 1440/256. Own full 277-column
  calibration, common incidence divisor and 716-pixel diagnostic support.
- Fixed 5% relative SVD cutoff, gain 0.15 and one-frame causal integrator;
  applied commands logged before the next update. All loops start at zero.
- Feedback uses noiseless photon-rate slopes. This is a deterministic Julia
  control fixture, not pyRTC, the operational 1296-signal mask, a noisy camera
  controller, transport, or hardware validation.
- Common 480² science grid computes piston-removed OPD RMS and exact coherent
  on-axis Strehl at 550 nm; matched-command checks at 960² and 1440² bound
  observed science quadrature sensitivity. Sparse same-command fine-sensor
  snapshots distinguish sensor error from divergent feedback trajectories.
- Additional 1920/256 and 1440/512 trajectories at seed 1, r0=0.15 m use the
  same 1440/256 calibration: plant sensitivity checks, not recalibrated anchors.

## Limits and decision

This study holds the native atmosphere's finite spatial spectrum fixed. The
3.66 m canvas omits spatial frequencies below 0.273 m⁻¹, substantially above
1/L0=0.033 m⁻¹, and 0.6 s winds traverse up to 6 m, reusing periodic structure.
It does not restore missing low-order power, constitute an independent
atmospheric or PWFS oracle, or validate infinite-screen/closed-loop statistics.
The focal spacing is also fixed by the maintained detector geometry. Compare
actual measured OPD distributions and per-seed results, not nominal r0 alone.

No physical accuracy tolerance was supplied. Report candidate/reference
errors, paired changes against the current preset, seed dispersion, and finer
reference sensitivities. Do not promote passing this finite fixture to an
unconditional accuracy-preserving default, particularly given earlier
high-frequency failures. Numerical and implementation improvements remain
separate: AOS profiling holds the existing prescription fixed.

## Timing

The first multi-seed process completed all 15 seed-1 runs, then ran out of
GPU memory preparing the next fine-reference dense modulation-phase array.
Its failure log is retained. Remaining seeds and each optional anchor run in
separate Julia processes; numerical harness source and protocol are unchanged.
This is process isolation for preparation memory, not evidence of a package
memory leak.

All GPU jobs are serialized. One Julia compute thread, no interactive pool,
one GC thread, BLAS/FFT 1, CPU affinity 2, uncontrolled shared-host GPU clocks.
Plant-only static warmed captured timings exclude screen generation/rendering,
uploads, host controller and science diagnostics. Candidates have 32 warmups,
128 timings and 16 allocation observations per run; costly references have
4 warmups, 16 timings and 16 allocation observations. These are sequential
service times, not fixed-rate deadline claims or complete control-loop cost.

A separate AOS harness measures the actual native five-layer/noisy detector
workload with completed stream/captured totals and serialized per-node
attribution; diagnostic timings are not additive, and profiling is separate
from uninstrumented timings.
