# REVOLTCopperSim.jl

`REVOLTCopperSim.jl` is the instrument-level REVOLT Copper simulation built on
[AdaptiveOpticsSim.jl](../AdaptiveOpticsSim.jl). It owns the modulated Pyramid
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

The opt-in process test sends this package's 64×64 Pyramid frames to an
independent pyRTC process, verifies pyRTC's 1,296-element signal geometry,
probes five independent HSDM277 directions, and processes an evolved
atmospheric frame. Install the pinned official pyRTC revision into an isolated
Python environment, then run:

```bash
python3 -m venv .venv-pyrtc
.venv-pyrtc/bin/python -m pip install --upgrade pip
.venv-pyrtc/bin/python -m pip install -r test/pyrtc/requirements.txt

export PYRTC_PYTHON="$PWD/.venv-pyrtc/bin/python"
REVOLT_COPPER_PYRTC_TESTS=1 julia --startup-file=no --project=. \
  -e 'using Pkg; Pkg.test()'
```

The Python environment is not part of the package runtime. The pyRTC test is
off by default because even its five-direction probe executes the complete
480-sample modulated Pyramid propagation. A qualified full-matrix gate remains
future work because the current HSDM influence and Copper reconstructor are
explicitly provisional.
