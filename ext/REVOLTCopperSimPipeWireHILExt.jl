module REVOLTCopperSimPipeWireHILExt

using REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.AlgorithmGraphs: prepared_graph_node
using AdaptiveOpticsSim.Detectors: detector_acquisition_plan
using AdaptiveOpticsSimPipeWireHIL
using PipeWireAO: SPA

function REVOLTCopperSim.pipewire_configuration(
    system::NamedTuple,
    remote::AbstractString;
    frame_node_name::AbstractString="revolt-copper-sim-wfs",
    command_node_name::AbstractString="revolt-copper-sim-command",
    timeout_ns::Integer=5_000_000_000,
    frame_encoding::Symbol=:float32,
    frame_schema::Union{Nothing,AbstractString}=nothing,
    command_schema::Union{Nothing,AbstractString}=nothing,
    command_scale::Real=1f0,
)
    graph, boundary = system.graph, system.boundary
    graph === boundary.graph || throw(ArgumentError(
        "system graph and HIL boundary must belong to the same prepared plant"))
    graph_name(graph) in (:revolt_copper_hil_grid_gaussian,
        :revolt_copper_hil_coordinate_gaussian) || throw(ArgumentError(
        "expected a maintained REVOLT Copper HIL graph"))
    detector = prepared_graph_node(graph, :detector)
    pdm = prepared_graph_node(graph, :pdm)
    atmosphere = prepared_graph_node(graph, :atmosphere)
    plan = detector_acquisition_plan(detector.prepared)
    boundary.command_input === graph_input(graph, :pdm_command) === pdm.pdm_command ||
        throw(ArgumentError("HIL command must bind the REVOLT HSDM277 input"))
    boundary.frame_output === graph_output(graph, :pwfs_frame) === detector.output ||
        throw(ArgumentError("HIL frame must bind the REVOLT Copper detector output"))
    size(hil_command_buffer(boundary)) == (command_count(),) &&
        eltype(hil_command_buffer(boundary)) === Float32 || throw(ArgumentError(
        "REVOLT HSDM277 commands require 277 Float32 elements"))
    size(hil_frame_buffer(boundary)) == plan.output_shape == (64, 64) &&
        eltype(hil_frame_buffer(boundary)) === Float32 || throw(ArgumentError(
        "REVOLT Copper frames require a 64×64 Float32 detector"))

    scientific_frame_schema = "org.revolt.copper.pwfs-frame.f32/1"
    metre_command_schema = "org.revolt.hsdm277.actuator-surface-opd-m.f32/1"
    # Existing REVOLT operational deployment contracts: raw ADC pixels and
    # demanded HSDM277 surface-OPD commands in micrometres.
    adc_frame_schema = "org.calculon.ao.raw-detector-pixels/1"
    micrometre_command_schema = "org.calculon.ao.demanded-pdm-command/1"
    if frame_encoding === :float32
        frame_schema === nothing && (frame_schema = scientific_frame_schema)
        frame_schema == scientific_frame_schema || throw(ArgumentError(
            "Float32 transport must preserve the scientific frame schema"))
    elseif frame_encoding === :uint16
        frame_schema == adc_frame_schema ||
            throw(ArgumentError("UInt16 ADC transport requires the explicit operational raw-detector-pixels schema"))
    else
        throw(ArgumentError("frame_encoding must be :float32 or :uint16"))
    end
    command_schema === nothing && (command_schema = metre_command_schema)
    ((command_schema == metre_command_schema && command_scale == 1) ||
     (command_schema == micrometre_command_schema && Float32(command_scale) == 1f-6)) ||
        throw(ArgumentError("command schema and scale must declare metre OPD (1) or micrometre OPD (1f-6)"))

    # Nanosecond quantization removes Float32 representation error without
    # assuming an integer-Hz cadence or substituting detector exposure for it.
    period_ns = round(Int, Float64(atmosphere.atmosphere_step) * 1e9)
    period_ns > 0 || throw(ArgumentError("atmosphere step must span a positive nanosecond period"))
    divisor = gcd(1_000_000_000, period_ns)
    return PipeWireHILConfiguration(;
        remote, frame_node_name, command_node_name, timeout_ns,
        frame_schema, command_schema, frame_encoding, command_scale,
        rate=SPA.Fraction(div(1_000_000_000, divisor), div(period_ns, divisor)),
        exposure_duration_ns=round(Int, Float64(plan.detector_params.exposure_duration) * 1e9),
    )
end

end
