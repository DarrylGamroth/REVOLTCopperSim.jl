# AOS implementation review for Copper CUDA

Read-only source review at AOS 7e6ea87ae2a6bf6c33569058ea1c09f4f4df839d. No GPU jobs, production changes, or new measurements. Source conclusions are observed static behavior; performance benefits below are hypotheses until profiled. Historical numbers come exclusively from REVOLTCopperSim.jl/benchmarks/results/2026-10-21-copper-convergence/README.md.

## Existing efficient behavior

Copper src/graphs.jl:113 selects PyramidShiftedMaskStrategy. AOS src/wfs/pyramid/optics.jl:627–662 computes one forward FFT of the common pupil field, then one backward FFT per modulation point through batched plans. It already avoids repeating the forward FFT for each tilt. src/wfs/pyramid/setup.jl:360–363,418–438 uses separable axis factors for unrotated, zero-rooftop paired masks. src/wfs/pyramid/setup.jl:188–202 reuses each focal sample within a thread across points. Replacing this with full 3D mask storage or claiming a new shared-forward-FFT optimization would duplicate/regress existing work.

Captured graph execution src/algorithm_graphs/captured_execution.jl:154–182 captures the complete enqueue path once; replay :502–510 launches the prepared device graph. Its host wrapper does not walk all optical nodes on each replay. Inspect device-node costs before rewriting the graph executor.

## Confirmed allocation inefficiency: dense pupil tilts retained by shifted-mask preparation

Observed source behavior: `src/wfs/pyramid/setup.jl:778–783` always calls `prepare_focal_plane_modulation` before dispatching on modulation propagation strategy. `src/optics/focal_plane_modulation.jl:156–174` allocates and fills a backend `resolution × resolution × point_count` Complex array, an equally sized host temporary, and copies the latter to the former. `PyramidOpticalFrontEnd` retains the prepared modulation, and `PyramidOpticsPlan` also binds that owner (`src/wfs/pyramid/stages.jl:172`), so the device phase array remains owned after preparation. The host array is preparation scratch and becomes eligible for reclamation after return; it is not retained scientific state.

Derived memory: 1440² × 256 × sizeof(ComplexF32) = 4,246,732,800 bytes = 3.955 GiB for the device phases, plus another 3.955 GiB host temporary during preparation. At production 480/32 the device array is 58,982,400 bytes = 56.25 MiB. These allocations are additional to the bounded batched FFT field stack, FFT workspaces, and propagation fields. The 256 MiB batch heuristic bounds only `field_stack`, not total preparation or retained modulation storage.

Runtime consumption is strategy dependent. Shifted-mask preparation reads only modulation phase-array dimensions/count, policy offsets and amplitude weights (`src/wfs/pyramid/setup.jl:400–422`); it derives the small offset vectors and separable factors or shifted masks. Ordinary shifted-mask rate propagation for both pupil and electric-field input uses the unmodulated focal field and these masks/factors, then returns before the dense-phase fallback (`src/wfs/pyramid/optics.jl:535–545,575–585`; callers `stages.jl:254–265,287–295`). Captured shifted-mask rate enqueue (`optics.jl:627–662`) likewise never consumes the dense phase elements. Thus the array's physical values are unused by the selected Copper rate-cycle execution even though its dimensions and target are currently used as metadata.

Scope matters: the array is not globally dead storage. `modulation_point_count(prepared)` derives count from its third dimension (`focal_plane_modulation.jl:124–125`), prepared input validation derives resolution from it (`stages.jl:157`), exact-target validation checks its backend (`src/wfs/exact_target.jl:433–440`), pupil-tilt reference propagation reads it, and public `pyramid_modulation_frame!` reads all its elements even on a sensor configured for shifted-mask rate execution (`optics.jl:774–812`). Different prepared modulation supplied to legacy intensity methods can also select the dense-phase fallback. Removing the allocation therefore needs strategy-specific ownership and direct updates to these call sites, not a zero-size placeholder or synthetic phase-array adapter.

Proposed remediation, not implemented: prepare a concrete shifted-mask modulation owner with explicit pupil resolution, sample count, policy, normalized amplitude weights and backend-bound offset/factor storage. Keep dense tilt storage in the pupil-tilt reference execution owner. Dispatch dimension/count/target validation on the actual representation. For the public modulation-camera product, prepare a separate execution owner that generates each required pupil tilt into reusable scratch, or use separable tilt factors without retaining N²M values. Preserve supported camera and alternate-modulation behavior through direct caller updates; do not add compatibility forwarding merely to mimic `.phases`. This is a memory/preparation optimization with unchanged physical discretization; it does not justify reducing N or M.

Current update API constraint: `update_cycle_averaged_circular_modulation!` validates against dense dimensions and regenerates only dense phases (`focal_plane_modulation.jl:227–244`, implementations `:182–215`). Shifted offset vectors are copied from the policy once (`setup.jl:416–422`) and mask/factor rebuilding consumes those vectors (`optics.jl:73–238`). No call from the update API updates those vectors. A strategy-specific design must explicitly define and implement radius/enable updates for each representation, preserve prepared addresses under capture, preserve point count/weights/origin and synchronize all products that share the scientific modulation. This source observation alone does not establish which external callers currently expect updates to affect a shifted rate owner.

Runtime evidence supplied by the primary investigation: after seed 1's 15 cases, preparing the next 1440/256 case failed with a 3.955 GiB CUDA allocation at `prepare_focal_plane_modulation` line 157. This matches the derived array size and identifies the immediate failed allocation. It does not establish a memory leak or explain all memory pressure: earlier fine-reference owners may remain reachable through benchmark loop scope, and memory-pool/FFT workspace residency also needs measurement. Running one process per seed is a harness isolation measure that preserves numerical source behavior, not a production fix.

Required validation before remediation: demonstrate retained device/peak host allocation reduction at fixed N/M; compare full detector rates, all 277 calibration columns and selected atmospheric trajectories against the current representation; verify the modulation-camera output and pupil-tilt reference path; exercise radius, phase-origin, enabled/disabled and sampled-weight behavior; verify capture/reset/replay, target validation and zero host allocation in repeated execution. Measure preparation separately from warmed execution. No GPU experiment or production edit was performed in this source trace.

## Candidate A: tune modulation batch size independently of discretization

Observed: src/wfs/pyramid/setup.jl:320–347 chooses the largest divisor of point_count bounded by 32 and 256 MiB of complex field planes. No device-specific timing enters the choice. At 1024² ComplexF32 and 32 points, the stack is exactly 256 MiB; at 128 points it is reused four times. Plan workspace is additional memory.

Hypothesis: smaller or larger device-qualified batches can improve FFT/cache/occupancy tradeoffs at unchanged optical sampling. Sweep divisors 1,2,4,8,16,32 (and only consider >32 after measuring memory) for fixed N/M, including FFT workspace and completed captured endpoint latency. Source changes would belong to preparation policy, never runtime autotuning. Also collect per-stage CUDA timings and memory traffic. The current value is a heuristic, not proof of an optimum. A change must preserve all weights, point ordering, bFFT normalization, exact-point coverage, zero host allocation, and captured replay/reset behavior.

## Candidate B: combine modulation-intensity reduction with detector binning

Observed: src/wfs/pyramid/optics.jl:645 zeroes a full padded intensity matrix; :657–660 reduces each stack into that full matrix. src/wfs/pyramid/setup.jl:74–85 reads the previous accumulated image and every complex stack plane; src/wfs/pyramid/stages.jl:338–354 then reads the completed padded image again to produce the small detector frame. Copper 480 uses a 1024² native map and 64² output, according to the archived configuration.

Hypothesis: a dedicated prepared no-LGS, detector-output execution seam can reduce abs2 and spatial cells directly into detector pixels, accumulating over modulation batches and eliminating native-intensity write/read passes and its clear. This cannot merely use 4096 threads each doing all work: each pixel covers 256 spatial samples × M points at 480, so parallel reduction structure matters. First low-risk alternative: let first modulation batch overwrite rather than read/add zeroed intensity, removing only clear/read overhead. Preserve the full native-map path where its product is actually observable or LGS convolution requires it. Validation: photon conservation, exact bin bounds, nonuniform amplitude/OPD, flat-subtracted slopes, interaction matrix, Float32 accumulation sensitivity, noisy frame sequence semantics, and captured HIL completion. Same discrete model should agree to an explicit numerical tolerance; atmospheric/closed-loop convergence does not substitute for this equivalence check.

## Candidate C: fuse the small fixed layer replay workload

Observed: src/atmosphere/multilayer.jl:712–773 launches one offset-advance kernel, one bilinear full-pupil interpolation/accumulation kernel per layer, then a separate pupil-mask pass. Five Copper layers imply seven kernels. Each layer after the first reads/writes the same OPD image. The first pass already overwrites output, so there is no redundant zero fill here. All offsets reside on device.

Hypothesis: prepared concrete tuple/family owner could interpolate all five screens per pixel into a register and apply pupil masking in one render kernel, preserving the separate offset advance. This reduces launch and intermediate-image traffic; additional register pressure/screen loads may erase the benefit. Do not replace concrete ownership with abstract arrays or Any. Validate frame 0 and long-run reset/replay equality, offset wrap, fractional interpolation, units, cn2 weights, pupil support and frozen-flow correlations. This affects the smaller Npupil grid, so prioritize only if measured stage timing warrants it.

## DM: distinguish the two profiles

Observed: src/optics/deformable_mirror.jl:36–53 coordinate Gaussian application evaluates exp for each illuminated pixel × command. At N=480, 277 commands that is up to 63.8 million exponentials before pupil exclusion. However Copper defaults to grid_gaussian (src/graphs.jl:223–233). Its captured path uses two separable contraction kernels with pupil masking fused into the second (AOS :1062–1086, kernels :66–91). The synchronous path uses two matrix multiplications and a mask kernel (:1042–1055), so profiling that public path does not establish captured-DM cost.

Hypotheses: only if measured as material, compare capture-safe fixed-shape BLAS or tiled contractions against current kernels. Preserve command mapping, misregistration and Gaussian width; do not truncate Gaussian support without an error budget. Dense materialization for coordinate profiles trades repeated transcendental work for large memory traffic/storage; it is not automatically better. Required check: all 277 central-difference columns plus arbitrary simultaneous commands, and matching captured versus stream results.

## Historical measurements and priorities

Archived completed noisy captured HIL medians: 480/32 2.654 ms; 180/64 0.803 ms; 480/128 9.877 ms. These establish discretization-dependent end-to-end costs, not individual kernel bottlenecks or benefits of any source change above. The same archive rejects universal 180/64 accuracy because of spatial aliasing and reports 480/32 modulation underconvergence for tested small tilts. Therefore do not call reduced pupil/modulation sampling an AOS implementation optimization. Keep physical/numerical convergence selection separate from same-model execution optimization.

Best next measurement: fixed production 480/32 and one accuracy-qualified candidate, captured timeline with forward FFT, backward FFT batches, stack fill, intensity reduction/binning, atmosphere, DM and HIL completion attributed. Then compare batch sizes and the smallest fusion change. No speedup estimate is justified without these measurements.

## Host/device preparation decision

The dense host phase array is a cold construction temporary, not a per-frame
transfer. Required phase products can be generated directly by backend kernels
and remain resident on device; this removes the large host temporary and upload,
but does not remove dense device storage. For the fixed shifted-mask rate path,
strategy-specific preparation can avoid the dense cube on both host and device,
retaining compact factors and explicit preparation for diagnostic/reference
products. Validate phase coordinates, quadrature, dtype, all strategies and
CPU/GPU agreement before changing this path. No production change is made here.
