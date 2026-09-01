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
