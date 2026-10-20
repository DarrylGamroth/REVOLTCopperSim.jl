module REVOLTCopperSimPipeWireHILExt

using REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.AlgorithmGraphs: prepared_graph_node
using AdaptiveOpticsSim.Detectors: detector_acquisition_plan
using AdaptiveOpticsSimPipeWireHIL
using AdaptiveOpticsSimPipeWireHIL: validate_hil_system, validate_wire_contract
using PipeWireAO: SPA

# Exact scientific and existing REVOLT deployment contracts. HEART is admitted
# only as its explicit paired raw-pixel/metre-OPD command contract.
const _WIRE_CONTRACTS = (
    (; frame_encoding=:float32, frame_schema=REVOLTCopperSim._FRAME_SCHEMA,
        command_schema=REVOLTCopperSim._PDM_COMMAND_SCHEMA, command_scale=1f0),
    (; frame_encoding=:float32, frame_schema=REVOLTCopperSim._FRAME_SCHEMA,
        command_schema="org.calculon.ao.demanded-pdm-command/1", command_scale=1f-6),
    (; frame_encoding=:uint16, frame_schema="org.calculon.ao.raw-detector-pixels/1",
        command_schema=REVOLTCopperSim._PDM_COMMAND_SCHEMA, command_scale=1f0),
    (; frame_encoding=:uint16, frame_schema="org.calculon.ao.raw-detector-pixels/1",
        command_schema="org.calculon.ao.demanded-pdm-command/1", command_scale=1f-6),
    (; frame_encoding=:uint16, frame_schema="org.heart.std-wfs.raw-pixels/1",
        command_schema="org.heart.std-dm.actuator-command/1", command_scale=1f0),
)

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
    validate_hil_system(system; command_input=:pdm_command,
        frame_output=:pwfs_frame, command_shape=(command_count(),), frame_shape=(64, 64))
    graph = system.graph
    detector = prepared_graph_node(graph, :detector)
    pdm = prepared_graph_node(graph, :pdm)
    plan = detector_acquisition_plan(detector.prepared)
    graph_input(graph, :pdm_command) === pdm.pdm_command || throw(ArgumentError(
        "HIL command must bind the REVOLT HSDM277 input"))
    graph_output(graph, :pwfs_frame) === detector.output || throw(ArgumentError(
        "HIL frame must bind the REVOLT Copper detector output"))
    plan.output_shape == (64, 64) || throw(ArgumentError(
        "REVOLT Copper requires a 64×64 detector plan"))

    frame_schema === nothing && frame_encoding === :float32 &&
        (frame_schema = REVOLTCopperSim._FRAME_SCHEMA)
    command_schema === nothing && (command_schema = REVOLTCopperSim._PDM_COMMAND_SCHEMA)
    validate_wire_contract(frame_encoding, frame_schema, command_schema,
        command_scale, _WIRE_CONTRACTS)

    # Nanosecond quantization removes Float32 representation error without
    # assuming an integer-Hz cadence or substituting detector exposure for it.
    frame_period_s = system.frame_period_s
    if hasproperty(graph.nodes, :atmosphere)
        atmosphere = prepared_graph_node(graph, :atmosphere)
        frame_period_s == atmosphere.atmosphere_step || throw(ArgumentError(
            "prepared frame period must match the atmosphere model-time step"))
    end
    period_ns = round(Int, Float64(frame_period_s) * 1e9)
    period_ns > 0 || throw(ArgumentError("prepared frame period must span a positive nanosecond period"))
    divisor = gcd(1_000_000_000, period_ns)
    return PipeWireHILConfiguration(;
        remote, frame_node_name, command_node_name, timeout_ns,
        frame_schema, command_schema, frame_encoding, command_scale,
        rate=SPA.Fraction(div(1_000_000_000, divisor), div(period_ns, divisor)),
        exposure_duration_ns=round(Int, Float64(plan.detector_params.exposure_duration) * 1e9),
    )
end

end
