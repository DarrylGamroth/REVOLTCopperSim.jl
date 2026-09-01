module REVOLTCopperSim

using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.Backends: HostComputeDevice

include("hsdm277.jl")
include("graphs.jl")

export actuator_coordinates
export actuator_grid_indices
export actuator_index_map
export command_count
export graph_path
export normalized_pupil_actuator_pitch
export prepare_calibration_system
export prepare_hil_system
export provisional_gaussian_influence_width
export provisional_mechanical_coupling
export supported_profiles

end # module REVOLTCopperSim
