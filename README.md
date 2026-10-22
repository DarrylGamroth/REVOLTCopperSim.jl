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

By default both retain the 480-sample pupil, 32-point shifted-mask Pyramid propagation,
64×64 detector boundary, and deterministic five-layer atmosphere. Neither is
a measured HSDM277 influence calibration or a qualified Copper pixel
reconstructor.

Instrument topology and the shared production/calibration prescription are
constructed in Julia in [`src/graphs.jl`](src/graphs.jl). Both deformable-mirror
profiles use the same scientific settings. TOML is reserved for experiment
settings and calibration references; graph topology is defined by the Julia
builders.

## Local use

The simulation uses registered AdaptiveOpticsSim 0.2.1 and test-only
AdaptiveOpticsCalibration 0.17. Add General and the package registry, then
instantiate this checkout with Julia 1.13:

```bash
git clone https://github.com/DarrylGamroth/REVOLTCopperSim.jl.git
cd REVOLTCopperSim.jl
julia --startup-file=no --project=. -e '
using Pkg
Pkg.Registry.add("General")
Pkg.Registry.add(Pkg.RegistrySpec(url="https://github.com/DarrylGamroth/PackageRegistry.git"))
Pkg.instantiate()'
```

```julia
using REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs

system = prepare_hil_system()
sequence = step_hil_frame!(system.boundary)
frame = hil_frame_buffer(system.boundary)
```

## Shared PipeWire HIL transport

The optional transport uses released `AdaptiveOpticsSimPipeWireHIL` 0.2.2
and `PipeWireAO`. Install those packages in the application environment, then
attach the shared adapter to the prepared plant:

```julia
using REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.Backends: HostComputeDevice
using AdaptiveOpticsSimPipeWireHIL
using PipeWireAO

system = REVOLTCopperSim.prepare_hil_system(;
    target=HostComputeDevice(), execution=StreamGraphExecution())
configuration = REVOLTCopperSim.pipewire_configuration(system, "rtc-private";
    frame_node_name="revolt-copper-wfs", command_node_name="revolt-copper-command")
transport = AdaptiveOpticsSimPipeWireHIL.prepare_pipewire_hil(system.boundary, configuration)
# The application owns start!(transport), exchange_frame!(transport), and close(transport).
```

The defaults publish the 64×64 Float32 scientific detector schema
`org.revolt.copper.pwfs-frame.f32/1` and receive 277 Float32 HSDM277
surface-OPD commands in metres. The atmosphere advances by 2 ms per frame
(500 Hz); the detector exposure is 2 ms. Configuration reads these
values from the prepared context and detector plan and checks its boundary bindings. It does not
rebuild the plant. Native, HEART, and pyRTC controllers connect through the
ordinary PipeWireAO routes using compatible schemas, independently of this
plant package.

Operational UInt16 ADC frames require an explicit `frame_encoding=:uint16`
and `frame_schema="org.calculon.ao.raw-detector-pixels/1"`. Micrometre OPD commands
require both `command_schema="org.calculon.ao.demanded-pdm-command/1"`
and `command_scale=1f-6`; conversion is confined to transport. Float32 frame
schemas and metre commands remain the default scientific contract.

The existing REVOLT HEART deployment contract is selected explicitly with
`frame_encoding=:uint16`, `frame_schema="org.heart.std-wfs.raw-pixels/1"`,
`command_schema="org.heart.std-dm.actuator-command/1"`, and `command_scale=1`.
The pair retains that deployment's metre-OPD command meaning and HSDM277 order;
other HEART command meanings require a separately validated conversion.

The optional configuration checks run in their own dependency environment:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=test/pipewire \
  -e 'using Pkg; Pkg.instantiate(); include("test/pipewire/runtests.jl")'
```

## Validation

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --startup-file=no --project=. \
  -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The default suite includes physical graph checks and pure Julia AOC calibration
contract tests. It does not launch Python or require a pyRTC installation.
The plant-only HIL graphs connect to an external RTC through
AdaptiveOpticsSimPipeWireHIL; REVOLTRTC supplies the shared Classic/Copper
PipeWireAO deployment and the FGA/JFG controller graph.

The opt-in process test calibrates this package's complete 277-command model
through its 64×64 Pyramid frames. It retains the complete interleaved
positive-poke/negative-poke response cycle, forms the 1,296×277 interaction
matrix with the test-only `AdaptiveOpticsCalibration.jl` dependency, and prepares an
identity-weighted, zero-Tikhonov `StrokeWeightedTikhonov` reconstructor. The
existing 5% singular-value cutoff selects its retained rank (221 for the
maintained fixture). The test compares both AOC products with the direct
central-difference matrix and pyRTC's existing cutoff pseudoinverse, then installs the
AOC corrector-by-measurement control matrix directly in an independent pyRTC
process before closing the unchanged 300-frame evolving-atmosphere loop. It
verifies retained interaction rank, reconstructor conditioning, on-axis Strehl
improvement, and pupil-OPD reduction. Install the pinned official pyRTC
revision into an isolated Python environment, then run:

```bash
python3 -m venv .venv-pyrtc
.venv-pyrtc/bin/python -m pip install --upgrade pip
.venv-pyrtc/bin/python -m pip install -r test/pyrtc/requirements.txt

export PYRTC_PYTHON="$PWD/.venv-pyrtc/bin/python"
REVOLT_COPPER_PYRTC_TESTS=1 julia --startup-file=no --project=. \
  -e 'using Pkg; Pkg.test()'
```

For the standalone metrics runner, use the committed test environment because
the AOC integration is deliberately test-only:

```bash
PYRTC_PYTHON="$PWD/.venv-pyrtc/bin/python" \
  julia --startup-file=no --project=test -e \
  'using Pkg; Pkg.instantiate(); include("test/pyrtc/run_revolt_copper_hil.jl")'
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
execution. The default benchmark environment supports CPU and CUDA 6.4.2; use
`benchmark/amdgpu` for AMDGPU so the accelerator stacks resolve independently.
Set `REVOLT_BENCH_EXECUTION=stream` to make an explicit accelerator diagnostic
comparison. The harness defaults to 10 CPU warmup cycles and 200 accelerator
warmup cycles so accelerator measurements exclude clock ramp-up and
first-replay effects:

```bash
julia --startup-file=no --project=benchmark \
  -e 'using Pkg; Pkg.instantiate()'

julia --startup-file=no --project=benchmark/amdgpu \
  -e 'using Pkg; Pkg.instantiate()'

REVOLT_BENCH_BACKEND=cpu \
REVOLT_BENCH_SAMPLES=20 \
REVOLT_BENCH_RUNS=3 \
REVOLT_BENCH_OUTPUT=/tmp/revolt-copper-cpu.toml \
JULIA_NUM_THREADS=1 \
  julia --startup-file=no --project=benchmark benchmark/frame_service.jl

REVOLT_BENCH_BACKEND=amdgpu \
REVOLT_BENCH_SAMPLES=20 \
REVOLT_BENCH_RUNS=3 \
REVOLT_BENCH_OUTPUT=/tmp/revolt-copper-amdgpu.toml \
JULIA_NUM_THREADS=1 \
  julia --startup-file=no --project=benchmark/amdgpu benchmark/frame_service.jl
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

## Numerical discretization study

The [CUDA pupil/modulation convergence study](benchmarks/results/2026-10-21-copper-convergence/README.md)
compares photon rates, flat-subtracted I4Q slopes and the full 277-actuator
calibration response against finer references, with complete captured-graph and
HIL timings. Public preparation keeps the 480²/32-point prescription unless an explicit
numerical mode is selected.

`benchmarks/run_convergence.py` runs the serial campaign in an isolated Julia
benchmark environment. NumPy handles archived-array reductions; Matplotlib is
optional for figures. The report records the local AOS source requirement,
commands, reference sensitivities and spatial-frequency limitations. This study
does not qualify a new production default or a measured Copper calibration.

The [paired atmospheric and closed-loop follow-up](benchmarks/results/2026-10-21-copper-atmospheric/README.md)
uses common seeded five-layer screens at three turbulence strengths, retaining
300 causal feedback updates per run. It reports sensor errors, control
trajectories, finer-reference sensitivity and science-grid checks separately.
This finite, noiseless control fixture does not qualify the operational noisy
RTC or change the production sampling default.

### Numerical modes

Both `prepare_hil_system` and `prepare_calibration_system` accept `mode`:

| Mode | Pupil resolution | Modulation points | Intended use |
|---|---:|---:|---|
| `:default` | 480 | 32 | Existing prescription |
| `:fast` | 240 | 96 | Faster RTC development candidate |
| `:accurate` | 1440 | 256 | Finer numerical comparison reference |

For example, use `prepare_hil_system(; mode=:fast)` and
`prepare_calibration_system(; mode=:fast)`. Returned `mode`, `pupil_resolution`,
and `modulation_points` record the resolved choice. Mode is independent of the
Gaussian DM `profile`, compute target and execution policy. Changing mode
requires rebuilding the plant and its corresponding calibration.

The atmospheric study observed worst same-command slope error of 0.952% and
maximum absolute mean residual RMS difference of 0.100% for 240/96 against
1440/256 over its nine paired seed/strength cases. Its median plant service
time was 1.851 ms versus 2.564 ms for the existing 480/32 case on an RTX 3080.
These finite noiseless diagnostic-loop results do not qualify a noisy operational
RTC loop, fixed-rate deadlines, or all spatial-frequency inputs. In particular,
180/64 is not selected because the spatial-frequency study exposed aliasing.
All modes retain shifted-mask propagation and the provisional instrument model;
`:accurate` denotes the finer numerical reference, not an independent physical
oracle or a universal accuracy guarantee. The finite periodic-screen atmosphere
also lacks subharmonic restoration of low-frequency power.

The lightweight CPU selector `test/modes.jl` checks all three resolved presets
and rejects invalid modes through both public builders without preparing an
optical graph. On Julia 1.13, all six checks passed on 2026-10-21. This validates
selection only; complete numerical-mode graph execution requires separate
validation. The prepared system exposes the resolved sampling as metadata;
record it with calibration and experiment results.

The [local CUDA mode validation](benchmarks/results/2026-10-21-copper-modes/README.md)
exercised both public captured constructors for every mode with the compact
modulation implementation in AOS `9a51f458`. All six cases preserved their
277-command/64×64 boundary and had zero host allocations in completed HIL
service. Median production service times were 2.653 ms (`:default`), 1.922 ms
(`:fast`), and 238.142 ms (`:accurate`) on the RTX 3080. These exclude external
RTC calculations and transport. Accurate mode is intended for finer numerical
checks when that cost is acceptable, not a 500 Hz deadline claim.
