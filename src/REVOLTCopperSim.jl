"""Instrument-level REVOLT Copper simulation and external-RTC boundary."""
module REVOLTCopperSim

using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.Backends: HostComputeDevice, allocate_device_array

include("hsdm277.jl")
include("graphs.jl")
include("science_diagnostics.jl")

export actuator_coordinates
export actuator_grid_indices
export actuator_index_map
export closed_loop_on_axis_strehl
export closed_loop_psf
export command_count
export graph_path
export normalized_pupil_actuator_pitch
export open_loop_on_axis_strehl
export open_loop_psf
export prepare_calibration_system
export prepare_hil_system
export prepare_science_diagnostics
export provisional_gaussian_influence_width
export provisional_mechanical_coupling
export science_pupil_support
export supported_profiles
export update_science_diagnostics!

end # module REVOLTCopperSim
