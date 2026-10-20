"""Instrument-level REVOLT Copper simulation and external-RTC boundary."""
module REVOLTCopperSim

using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.Backends: HostComputeDevice, allocate_device_array
using AdaptiveOpticsSim.WavefrontSensors: PyramidShiftedMaskStrategy

include("parameters.jl")
include("geometry.jl")
include("graphs.jl")

export actuator_coordinates
export actuator_grid_indices
export actuator_index_map
export command_count
export normalized_pupil_actuator_pitch
export pipewire_configuration
export prepare_calibration_system
export prepare_hil_system
export provisional_gaussian_influence_width
export provisional_mechanical_coupling
export supported_profiles

"""
    pipewire_configuration(system, remote; kwargs...)

Configure the shared PipeWire HIL transport for an already prepared REVOLT Copper
plant. Load `AdaptiveOpticsSimPipeWireHIL` and `PipeWireAO` to enable this optional
method. Frame cadence comes from the prepared context and exposure from the named detector;
configuration neither rebuilds nor steps the plant. Defaults preserve the Float32
scientific detector frame and 277-element HSDM277 surface-OPD command in metres.

Endpoint keywords are `frame_node_name`, `command_node_name`, and `timeout_ns`.
UInt16 ADC transport requires `frame_encoding=:uint16` and an explicit distinct
`frame_schema="org.calculon.ao.raw-detector-pixels/1"`. Micrometre OPD commands
require the explicit `command_schema="org.calculon.ao.demanded-pdm-command/1"` with
`command_scale=1f-6`. The existing REVOLT HEART deployment requires its explicit
`frame_encoding=:uint16`, `frame_schema="org.heart.std-wfs.raw-pixels/1"`,
`command_schema="org.heart.std-dm.actuator-command/1"`, and `command_scale=1` pair.
Conversions apply only at the transport boundary.
"""
function pipewire_configuration end

end # module REVOLTCopperSim
