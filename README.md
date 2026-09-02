# REVOLTCopperSim.jl

`REVOLTCopperSim.jl` is the instrument-level REVOLT Copper simulation built on
[AdaptiveOpticsSim.jl](https://github.com/DarrylGamroth/AdaptiveOpticsSim.jl).
It owns the modulated Pyramid
sensor graph, EMCCD model, HSDM277 command geometry, external-RTC boundary, and
instrument-specific validation.

The package currently provides two explicitly provisional deformable-mirror
profiles:

- `:coordinate_gaussian` evaluates the supplied actuator coordinates;
- `:grid_gaussian` uses the equivalent separable 19×19 regular-grid model.

Both retain the 480-sample pupil, 32-point shifted-mask Pyramid propagation,
64×64 detector boundary, and deterministic five-layer atmosphere. Neither is
a measured HSDM277 influence calibration or a qualified Copper pixel
reconstructor.

## Local use

Until these development packages are registered, clone AOS and this package as
sibling directories so the checked-in `[sources]` entry resolves:

```bash
mkdir revolt-copper-work
cd revolt-copper-work
git clone https://github.com/DarrylGamroth/AdaptiveOpticsSim.jl.git
git clone https://github.com/DarrylGamroth/REVOLTCopperSim.jl.git
cd REVOLTCopperSim.jl
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
```

```julia
using REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs

system = prepare_hil_system()
sequence = step_hil_frame!(system.boundary)
frame = hil_frame_buffer(system.boundary)
```

## Validation

```bash
julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The opt-in process test calibrates this package's complete 277-command model
through its 64×64 Pyramid frames, installs the resulting 1,296×277
interaction matrix in an independent pyRTC process, and closes a 300-frame
evolving-atmosphere loop. It verifies retained interaction rank, reconstructor
conditioning, on-axis Strehl improvement, and pupil-OPD reduction. Install the
pinned official pyRTC revision into an isolated Python environment, then run:

```bash
python3 -m venv .venv-pyrtc
.venv-pyrtc/bin/python -m pip install --upgrade pip
.venv-pyrtc/bin/python -m pip install -r test/pyrtc/requirements.txt

export PYRTC_PYTHON="$PWD/.venv-pyrtc/bin/python"
REVOLT_COPPER_PYRTC_TESTS=1 julia --startup-file=no --project=. \
  -e 'using Pkg; Pkg.test()'
```

The Python environment is not part of the package runtime. The pyRTC test is
off by default because its full interaction matrix and atmospheric loop execute
the complete 480-sample, 32-point modulated Pyramid propagation. This validates
the self-consistency of the maintained simulated instrument; the HSDM influence
model and resulting Copper reconstructor remain explicitly provisional rather
than measured instrument calibrations.

With the pinned pyRTC revision and deterministic SplitMix64 atmosphere seed,
the maintained gate retains 221 of 277 interaction directions at its 5%
singular-value cutoff. Over 300 atmospheric frames, the steady-state mean
on-axis Strehl increases from 0.0208 to 0.437 and pupil OPD RMS decreases from
290 nm to 79.8 nm. These values are regression references for this simulated
instrument, not measured REVOLT Copper performance.

## Frame-service benchmark

The package benchmark measures the serialized HIL service boundary: one
complete evolving-atmosphere graph step through a host-visible detector frame,
followed by immediate adoption of a zero RTC command. RTC computation and
fixed-arrival queueing are intentionally excluded. Run it with one Julia
thread; select `cpu`, `amdgpu`, or `cuda`. Accelerator runs default to one
captured device graph for the complete frame, while CPU runs use direct stream
execution. Set `REVOLT_BENCH_EXECUTION=stream` to make an explicit accelerator
diagnostic comparison. The harness defaults to 10 CPU warmup cycles and 200
accelerator warmup cycles so accelerator measurements exclude clock ramp-up
and first-replay effects:

```bash
julia --startup-file=no --project=benchmark \
  -e 'using Pkg; Pkg.instantiate()'

REVOLT_BENCH_BACKEND=cpu \
REVOLT_BENCH_SAMPLES=20 \
REVOLT_BENCH_RUNS=3 \
REVOLT_BENCH_OUTPUT=/tmp/revolt-copper-cpu.toml \
JULIA_NUM_THREADS=1 \
  julia --startup-file=no --project=benchmark benchmark/frame_service.jl
```

The TOML artifact records raw samples, preparation, warmed Julia allocation,
p50/p90 (and p99 only with at least 100 samples), mean frame and cycle rates,
source revisions, dirty state, runtime versions, hardware, affinity, and power
policy. These are self-paced service-cost measurements, not fixed-rate
deadline claims.

### Recorded results (2026-09-01)

| Backend | Execution | Mean frame rate | Mean HIL cycle rate | Frame p50 / p90 / p99 | Warmed Julia bytes/cycle |
|---|---|---:|---:|---:|---:|
| CPU | stream | 3.370 frames/s | 3.370 cycles/s | 293.157 / 318.147 / 339.881 ms | 0 |
| AMDGPU | captured HIP graph | 19.756 frames/s | 19.741 cycles/s | 49.958 / 52.628 / 53.632 ms | 0 |
| CUDA | captured CUDA graph | 76.781 frames/s | 76.165 cycles/s | 12.989 / 13.431 / 17.995 ms | 0 |

Each row contains three fresh prepared runs of 100 measured frames. CPU used
ten warmup frames per run; both accelerator results used 200. The
[CPU](benchmark/results/2026-09-01-cpu.toml),
[AMDGPU](benchmark/results/2026-09-01-amdgpu.toml), and
[CUDA](benchmark/results/2026-09-01-cuda.toml) artifacts contain the raw
samples and provenance.

The earlier CUDA artifact used CUDA.jl's task-coordinated stream wait at every
captured completion boundary. AdaptiveOpticsSim `aa3d934` selects direct
blocking stream completion only for capture-qualified execution, which cannot
contain host callbacks. The corrected row changes that completion mechanism;
the instrument graph and timed service boundary are unchanged.

AdaptiveOpticsSim `5c17514` also forms each fixed-modulation Pyramid batch with
one work item per focal coordinate, loading the shared focal field once while
writing all 32 planes. The optical model, FFT count, modulation sampling, and
output shape are unchanged. The CUDA result showed run-to-run laptop power
variation; its raw artifact retains the slower third run and resulting p99.

CPU and AMDGPU ran on `rtc-devel` with an AMD Ryzen 7 6800H and its integrated
Rembrandt GPU. CUDA ran under WSL2 on `DGAMROTH-XPS` with an Intel Core
i7-12700H and RTX 3050 Ti Laptop GPU. These measurements therefore establish
the service cost on the available systems; they are not an isolated
AMD-versus-NVIDIA hardware comparison. Each process was pinned to one CPU, but
the recorded system load was not otherwise quiescent.
