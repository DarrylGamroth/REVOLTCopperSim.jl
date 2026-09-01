module PyRTCProcessHIL

using AdaptiveOpticsSim.AlgorithmGraphs
using LinearAlgebra
import REVOLTCopperSim

include(joinpath(@__DIR__, "pyrtc_shared_memory.jl"))
using .PyRTCSharedMemory

const PYRTC_STREAM_NAMES = ("wfs", "wfc", "signal", "signal2D")
const WORKER_PREFIX = "REVOLT_COPPER_PYRTC_WORKER "

struct PyRTCProcessDefinition
    name::Symbol
    wavefront_sensor::Symbol
    frame_shape::Tuple{Int,Int}
    signal_shape::Tuple{Int}
    signal_2d_shape::Tuple{Int,Int}
    command_count::Int
    poke::Float32
end

@inline process_definition() = PyRTCProcessDefinition(
    :revolt_copper,
    :pyramid,
    (64, 64),
    (1296,),
    (30, 60),
    REVOLTCopperSim.command_count(),
    2.0f-8,
)

@inline prepare_calibration_system(::PyRTCProcessDefinition) =
    REVOLTCopperSim.prepare_calibration_system()
@inline prepare_atmospheric_system(::PyRTCProcessDefinition) =
    REVOLTCopperSim.prepare_hil_system()

struct ProcessStreams{W,C,S,D}
    wfs::W
    wfc::C
    signal::S
    signal_2d::D
end

mutable struct PyRTCWorker
    process::Base.Process
    stopped::Bool
end

function require_available_stream_names()
    Sys.islinux() || error("the pyRTC shared-memory integration requires Linux")
    occupied = String[]
    for name in PYRTC_STREAM_NAMES, suffix in ("", "_meta", "_gpu_handle")
        path = joinpath("/dev/shm", name * suffix)
        ispath(path) && push!(occupied, path)
    end
    isempty(occupied) || error(
        "refusing to reuse active pyRTC shared-memory streams: " *
        join(occupied, ", "),
    )
    return nothing
end

function create_process_streams(definition::PyRTCProcessDefinition)
    require_available_stream_names()
    wfs = nothing
    wfc = nothing
    signal = nothing
    signal_2d = nothing
    try
        wfs = create_stream("wfs", Float32, definition.frame_shape)
        wfc = create_stream("wfc", Float32, (definition.command_count,))
        signal = create_stream("signal", Float32, definition.signal_shape)
        signal_2d = create_stream(
            "signal2D",
            Float32,
            definition.signal_2d_shape,
        )
        return ProcessStreams(wfs, wfc, signal, signal_2d)
    catch
        !isnothing(signal_2d) && close_and_unlink_noexcept!(signal_2d)
        !isnothing(signal) && close_and_unlink_noexcept!(signal)
        !isnothing(wfc) && close_and_unlink_noexcept!(wfc)
        !isnothing(wfs) && close_and_unlink_noexcept!(wfs)
        rethrow()
    end
end

function close_and_unlink_noexcept!(stream::PyRTCStream)
    try
        close(stream)
    catch
    end
    try
        unlink!(stream)
    catch
    end
    return nothing
end

function close_process_streams!(streams::ProcessStreams)
    close_and_unlink_noexcept!(streams.signal_2d)
    close_and_unlink_noexcept!(streams.signal)
    close_and_unlink_noexcept!(streams.wfc)
    close_and_unlink_noexcept!(streams.wfs)
    return nothing
end

function pyrtc_python()
    executable = get(
        ENV,
        "PYRTC_PYTHON",
        get(
            ENV,
            "JULIA_PYTHONCALL_EXE",
            something(Sys.which("python3"), ""),
        ),
    )
    isfile(executable) || error(
        "set PYRTC_PYTHON to the Python interpreter containing pyRTC dependencies",
    )
    return abspath(executable)
end

@inline _worker_valid_subapertures_path(
    ::Val{:shack_hartmann},
    ::AbstractString,
) = nothing
@inline _worker_valid_subapertures_path(
    ::Val{:pyramid},
    ::AbstractString,
) = nothing

@inline function _worker_valid_subapertures_path(
    ::Val{:revolt_copper},
    ::AbstractString,
)
    return nothing
end

function worker_command(
    definition::PyRTCProcessDefinition,
    temporary_directory::AbstractString,
)
    arguments = String[
        pyrtc_python(),
        joinpath(@__DIR__, "pyrtc_process_worker.py"),
        "--system",
        String(definition.name),
        "--temporary-directory",
        abspath(temporary_directory),
    ]
    valid_subapertures_path = _worker_valid_subapertures_path(
        Val(definition.name),
        temporary_directory,
    )
    if !isnothing(valid_subapertures_path)
        append!(arguments, (
            "--valid-subapertures-file",
            abspath(valid_subapertures_path),
        ))
    end
    return Cmd(arguments)
end

function read_worker_message(process::Base.Process)
    while !eof(process)
        line = readline(process)
        startswith(line, WORKER_PREFIX) || continue
        return chop(line; head=length(WORKER_PREFIX), tail=0)
    end
    error("pyRTC worker exited without a control response")
end

function await_worker_message(
    worker::PyRTCWorker;
    timeout::Real=10.0,
)
    task = @async read_worker_message(worker.process)
    status = timedwait(() -> istaskdone(task), timeout; pollint=0.001)
    if status === :timed_out
        process_running(worker.process) && kill(worker.process)
        try
            wait(worker.process)
        catch
        end
        worker.stopped = true
        try
            wait(task)
        catch
        end
        error("pyRTC worker did not respond within $timeout seconds")
    end
    return fetch(task)
end

function start_worker(
    definition::PyRTCProcessDefinition,
    temporary_directory::AbstractString,
)
    process = open(
        worker_command(definition, temporary_directory),
        "r+",
    )
    worker = PyRTCWorker(process, false)
    try
        ready = split(await_worker_message(worker; timeout=30.0))
        length(ready) == 3 && first(ready) == "READY" || error(
            "unexpected pyRTC worker startup response: $(join(ready, " "))",
        )
        parse(Int, ready[2]) == only(definition.signal_shape) || error(
            "pyRTC worker signal length $(ready[2]) does not match " *
            "$(only(definition.signal_shape))",
        )
        parse(Int, ready[3]) == definition.command_count || error(
            "pyRTC worker command length $(ready[3]) does not match " *
            "$(definition.command_count)",
        )
        return worker
    catch
        stop_worker_noexcept!(worker)
        rethrow()
    end
end

function send_worker_command!(
    worker::PyRTCWorker,
    command::AbstractString,
    expected_response::AbstractString;
    timeout::Real=10.0,
)
    worker.stopped && error("pyRTC worker is stopped")
    write(worker.process, command, '\n')
    flush(worker.process)
    response = await_worker_message(worker; timeout)
    response == expected_response || error(
        "pyRTC worker responded '$response'; expected '$expected_response'",
    )
    return nothing
end

function stop_worker!(worker::PyRTCWorker)
    worker.stopped && return nothing
    if process_running(worker.process)
        try
            send_worker_command!(worker, "STOP", "STOPPED"; timeout=5.0)
        catch
            process_running(worker.process) && kill(worker.process)
        end
    end
    wait(worker.process)
    worker.stopped = true
    success(worker.process) || error("pyRTC worker exited unsuccessfully")
    return nothing
end

function stop_worker_noexcept!(worker::PyRTCWorker)
    worker.stopped && return nothing
    if process_running(worker.process)
        try
            write(worker.process, "STOP\n")
            flush(worker.process)
        catch
        end
        timedwait(
            () -> !process_running(worker.process),
            1.0;
            pollint=0.01,
        )
        process_running(worker.process) && kill(worker.process)
    end
    try
        wait(worker.process)
    catch
    end
    worker.stopped = true
    return nothing
end

function process_frame!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    frame::AbstractMatrix{Float32},
    signal::Vector{Float32};
    command::AbstractString="PROCESS",
    response::AbstractString="PROCESSED",
)
    publish!(streams.wfs, frame)
    send_worker_command!(worker, command, response)
    read_next!(signal, streams.signal; timeout=5.0)
    return signal
end

function set_flat_reference!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    prepared,
    signal::Vector{Float32},
)
    boundary = prepared.boundary
    sequence = step_hil_frame!(boundary)
    process_frame!(
        worker,
        streams,
        hil_frame_buffer(boundary),
        signal,
    )
    send_worker_command!(worker, "SET_REF", "REF_SET")
    process_frame!(
        worker,
        streams,
        hil_frame_buffer(boundary),
        signal,
    )
    return sequence, copy(signal)
end

function calibrate_interaction_matrix!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    prepared,
    signal::Vector{Float32};
    poke::Float32,
    report_progress::Bool=false,
)
    boundary = prepared.boundary
    sequence, flat_signal = set_flat_reference!(
        worker,
        streams,
        prepared,
        signal,
    )
    interaction_matrix = Matrix{Float32}(
        undef,
        length(signal),
        length(hil_command_buffer(boundary)),
    )
    positive_signal = similar(signal)

    for command_index in axes(interaction_matrix, 2)
        if report_progress && (command_index == 1 ||
                command_index % 25 == 0 ||
                command_index == last(axes(interaction_matrix, 2)))
            println(
                "  calibrated command ",
                command_index,
                " / ",
                size(interaction_matrix, 2),
            )
        end
        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        copyto!(
            positive_signal,
            process_frame!(
                worker,
                streams,
                hil_frame_buffer(boundary),
                signal,
            ),
        )

        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = -poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        negative_signal = process_frame!(
            worker,
            streams,
            hil_frame_buffer(boundary),
            signal,
        )
        @views @. interaction_matrix[:, command_index] =
            (positive_signal - negative_signal) / (2 * poke)
    end

    fill!(hil_command_buffer(boundary), 0.0f0)
    adopt_hil_command!(boundary, sequence)
    reset_hil_boundary!(boundary)
    return flat_signal, interaction_matrix
end

function probe_interaction_columns!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    prepared,
    signal::Vector{Float32},
    command_indices::NTuple{N,Int};
    poke::Float32,
) where {N}
    boundary = prepared.boundary
    all(index -> index in eachindex(hil_command_buffer(boundary)),
        command_indices) || throw(BoundsError(
        hil_command_buffer(boundary),
        command_indices,
    ))
    sequence, flat_signal = set_flat_reference!(
        worker,
        streams,
        prepared,
        signal,
    )
    interaction_columns = Matrix{Float32}(undef, length(signal), N)
    positive_signal = similar(signal)

    for (column_index, command_index) in pairs(command_indices)
        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        copyto!(
            positive_signal,
            process_frame!(
                worker,
                streams,
                hil_frame_buffer(boundary),
                signal,
            ),
        )

        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = -poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        negative_signal = process_frame!(
            worker,
            streams,
            hil_frame_buffer(boundary),
            signal,
        )
        @views @. interaction_columns[:, column_index] =
            (positive_signal - negative_signal) / (2 * poke)
    end

    fill!(hil_command_buffer(boundary), 0.0f0)
    adopt_hil_command!(boundary, sequence)
    reset_hil_boundary!(boundary)
    return flat_signal, interaction_columns
end

function run_revolt_copper_validation()
    definition = process_definition()
    calibration = prepare_calibration_system(definition)
    streams = create_process_streams(definition)
    worker = nothing
    command_indices = (70, 105, 139, 173, 208)
    return mktempdir() do temporary_directory
        try
            worker = start_worker(definition, temporary_directory)
            signal = zeros(Float32, definition.signal_shape)
            flat_signal, interaction_columns = probe_interaction_columns!(
                worker,
                streams,
                calibration,
                signal,
                command_indices;
                poke=definition.poke,
            )
            norm(flat_signal) <= 1.0f-5 || error(
                "REVOLT Copper flat reference left a nonzero residual: " *
                "$(norm(flat_signal))",
            )
            all(isfinite, interaction_columns) || error(
                "REVOLT Copper interaction probes contain a non-finite value",
            )
            response_norms = Tuple(
                norm(@view(interaction_columns[:, column_index]))
                for column_index in axes(interaction_columns, 2)
            )
            all(>(0), response_norms) || error(
                "REVOLT Copper interaction probes contain a zero response",
            )
            singular_values = svdvals(interaction_columns)
            rank_tolerance = maximum(singular_values) * 1.0f-4
            probe_rank = count(>(rank_tolerance), singular_values)
            probe_rank == length(command_indices) || error(
                "REVOLT Copper interaction probes have rank $probe_rank; " *
                "expected $(length(command_indices))",
            )

            atmospheric = prepare_atmospheric_system(definition)
            step_hil_frame!(atmospheric.boundary)
            process_frame!(
                worker,
                streams,
                hil_frame_buffer(atmospheric.boundary),
                signal,
            )
            all(isfinite, signal) || error(
                "REVOLT Copper atmospheric Pyramid signal is non-finite",
            )
            atmospheric_signal_norm = norm(signal)
            atmospheric_signal_norm > 0 || error(
                "REVOLT Copper atmospheric Pyramid signal is zero",
            )
            stop_worker!(worker)

            return (;
                system=definition.name,
                frame_shape=definition.frame_shape,
                signal_length=length(signal),
                command_count=definition.command_count,
                command_indices,
                response_norms,
                probe_rank,
                probe_condition=maximum(singular_values) /
                    minimum(singular_values),
                atmospheric_signal_norm,
            )
        finally
            !isnothing(worker) && stop_worker_noexcept!(worker)
            close_process_streams!(streams)
        end
    end
end

function revolt_copper_main()
    result = run_revolt_copper_validation()
    println("REVOLT Copper/native-SHM pyRTC process probe passed")
    println("  detector frame shape: ", result.frame_shape)
    println("  signal length: ", result.signal_length)
    println("  PDM command length: ", result.command_count)
    println("  probed commands: ", result.command_indices)
    println("  probe rank: ", result.probe_rank)
    println("  probe condition: ", result.probe_condition)
    println("  atmospheric signal norm: ", result.atmospheric_signal_norm)
    return nothing
end

end # module PyRTCProcessHIL
