# Copper numerical-mode CUDA validation — 2026-10-21

The public production and calibration builders were tested with local AOS
`9a51f458` (strategy-specific modulation storage), Julia 1.13.2, CUDA 6.4.2,
and an RTX 3080 10 GiB, driver 615.71.10. One Julia/BLAS/FFT thread, CPU affinity
2, shared host, uncontrolled GPU clocks. No RTC or transport was connected.

| Mode | Pupil / points | Calibration median | Production median | Modulation GPU storage |
|---|---|---:|---:|---:|
| default | 480 / 32 | 2.587 ms | 2.653 ms | 2176 B |
| fast | 240 / 96 | 1.877 ms | 1.922 ms | 1728 B |
| accurate | 1440 / 256 | 235.982 ms | 238.142 ms | 7808 B |

Every case used native captured execution and retained compact modulation.
Calibration capture consumed changed nonuniform OPD and reproduced flat OPD
exactly. Production atmosphere evolved and its seeded two-frame sequence
replayed exactly after reset. Actual pupil dimensions, point counts, resolved
metadata and 277-command/64×64-frame boundaries were checked. All 16 warmed
host-allocation observations per constructor were zero (96 total).

Timing boundary: completed `step_hil_frame!` plus matching `adopt_hil_command!`,
including host staging and graph completion. Each constructor had 32 warmups
and 128 samples. Raw samples, observed maxima, environment/source records and
checks are in `copper-modes.toml`; this single-process campaign is not a
worst-case latency or deadline guarantee. Preparation timings include cold
compilation and must not be compared as warmed preparation costs.

Run `benchmarks/copper_modes_cuda.jl OUTPUT.toml` in an isolated environment
containing CUDA, this Copper source, and the optimized AOS source. The archived
Project/Manifest files describe the validation environment; adjust checkout
paths when reproducing. Application environments retain their registered AOS
pins and need a release update to use the new storage implementation.

Mode selection is a numerical accuracy/cost choice. The existing default is
unchanged. The atmospheric convergence evidence and approximation limits are
linked from the maintained README; this execution check adds no independent
optical oracle or noisy closed-loop accuracy qualification.
