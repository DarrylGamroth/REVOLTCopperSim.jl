module PyRTCProcessHIL

using AdaptiveOpticsSim.AlgorithmGraphs
using LinearAlgebra
import REVOLTCopperSim

include(joinpath(@__DIR__, "pyrtc_shared_memory.jl"))
using .PyRTCSharedMemory

const PYRTC_STREAM_NAMES = ("wfs", "wfc", "signal", "signal2D")
const WORKER_PREFIX = "REVOLT_COPPER_PYRTC_WORKER "
const ATMOSPHERE_VALIDATION_BURN_IN_FRAMES = 10
# SplitMix64 seed 1 produces approximately 21× mean Strehl improvement for
# this deterministic fixture. Fifteen leaves a material regression margin.
const REVOLT_COPPER_MINIMUM_STREHL_IMPROVEMENT = 15.0

struct PyRTCProcessDefinition
    name::Symbol
    wavefront_sensor::Symbol
    frame_shape::Tuple{Int,Int}
    signal_shape::Tuple{Int}
    signal_2d_shape::Tuple{Int,Int}
    command_count::Int
    poke::Float32
    gain::Float32
    control_rcond::Float32
    iterations::Int
end

@inline process_definition() = PyRTCProcessDefinition(
    :revolt_copper,
    :pyramid,
    (64, 64),
    (1296,),
    (30, 60),
    REVOLTCopperSim.command_count(),
    2.0f-8,
    0.15f0,
    5.0f-2,
    300,
)

@inline prepare_calibration_system(::PyRTCProcessDefinition) =
    REVOLTCopperSim.prepare_calibration_system()
@inline prepare_atmospheric_system(::PyRTCProcessDefinition) =
    REVOLTCopperSim.prepare_hil_system()
@inline prepare_science_diagnostics() =
    REVOLTCopperSim.prepare_science_diagnostics()

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

function configure_worker_loop!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    interaction_matrix::Matrix{Float32},
    gain::Float32,
    control_rcond::Float32,
    temporary_directory::AbstractString,
)
    matrix_path = joinpath(temporary_directory, "interaction_matrix.f32")
    open(matrix_path, "w") do io
        write(io, interaction_matrix)
    end
    send_worker_command!(
        worker,
        "CONFIGURE $matrix_path $(Float64(gain)) " *
        "$(Float64(control_rcond))",
        "CONFIGURED";
        timeout=30.0,
    )
    send_worker_command!(worker, "FLATTEN", "FLATTENED")
    flat_command = zeros(Float32, size(interaction_matrix, 2))
    read_next!(flat_command, streams.wfc; timeout=5.0)
    all(iszero, flat_command) || error(
        "pyRTC worker flatten command is nonzero",
    )
    return nothing
end

@inline function root_mean_square(values::AbstractArray)
    return sqrt(sum(abs2, values) / length(values))
end

function pupil_opd_rms(
    opd::AbstractMatrix{<:AbstractFloat},
    support::AbstractMatrix{Bool},
)
    axes(opd) == axes(support) || throw(DimensionMismatch(
        "the pupil OPD and support must have identical axes",
    ))
    sample_count = 0
    mean_opd = 0.0
    sum_squared_difference = 0.0
    for index in eachindex(opd, support)
        support[index] || continue
        sample_count += 1
        value = Float64(opd[index])
        difference = value - mean_opd
        mean_opd += difference / sample_count
        sum_squared_difference += difference * (value - mean_opd)
    end
    sample_count > 0 || error("the HIL reference pupil support is empty")
    return sqrt(sum_squared_difference / sample_count)
end

function mean_from(values::Vector{<:Real}, first_index::Int)
    first_index in eachindex(values) || throw(BoundsError(values, first_index))
    total = 0.0
    sample_count = 0
    for index in first_index:lastindex(values)
        total += Float64(values[index])
        sample_count += 1
    end
    return total / sample_count
end

function close_atmospheric_loop!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    definition::PyRTCProcessDefinition,
    signal::Vector{Float32};
    frames::Int=definition.iterations,
)
    frames > ATMOSPHERE_VALIDATION_BURN_IN_FRAMES || throw(ArgumentError(
        "atmosphere validation requires more than " *
        "$(ATMOSPHERE_VALIDATION_BURN_IN_FRAMES) frames",
    ))
    prepared = prepare_atmospheric_system(definition)
    diagnostics = prepare_science_diagnostics()
    boundary = prepared.boundary
    command = zeros(Float32, definition.command_count)
    open_loop_values = Vector{Float32}(undef, frames)
    closed_loop_values = Vector{Float32}(undef, frames)
    uncompensated_opd_rms_values = Vector{Float64}(undef, frames)
    residual_opd_rms_values = Vector{Float64}(undef, frames)
    pdm_command_rms_values = Vector{Float64}(undef, frames)
    support = REVOLTCopperSim.science_pupil_support(diagnostics)
    sequence = step_hil_frame!(boundary)

    for frame_index in 1:frames
        atmosphere_opd = graph_output(prepared.graph, Val(:atmosphere_opd))
        residual_opd = graph_output(prepared.graph, Val(:pupil_opd))
        REVOLTCopperSim.update_science_diagnostics!(
            diagnostics,
            atmosphere_opd,
            residual_opd,
        )
        open_loop_values[frame_index] =
            REVOLTCopperSim.open_loop_on_axis_strehl(diagnostics)
        closed_loop_values[frame_index] =
            REVOLTCopperSim.closed_loop_on_axis_strehl(diagnostics)
        uncompensated_opd_rms_values[frame_index] =
            pupil_opd_rms(atmosphere_opd, support)
        residual_opd_rms_values[frame_index] =
            pupil_opd_rms(residual_opd, support)
        maximum(REVOLTCopperSim.open_loop_psf(diagnostics)) <= 1.001f0 || error(
            "open-loop PSF exceeds its exact diffraction-limited peak",
        )
        maximum(REVOLTCopperSim.closed_loop_psf(diagnostics)) <= 1.001f0 || error(
            "closed-loop PSF exceeds its exact diffraction-limited peak",
        )

        process_frame!(
            worker,
            streams,
            hil_frame_buffer(boundary),
            signal;
            command="STEP",
            response="STEPPED",
        )
        read_next!(command, streams.wfc; timeout=5.0)
        pdm_command_rms_values[frame_index] = root_mean_square(command)
        copyto!(hil_command_buffer(boundary), command)
        adopt_hil_command!(boundary, sequence)
        frame_index < frames && (sequence = step_hil_frame!(boundary))
    end

    first_steady_frame = ATMOSPHERE_VALIDATION_BURN_IN_FRAMES + 1
    mean_open_loop_on_axis_strehl =
        mean_from(open_loop_values, first_steady_frame)
    mean_closed_loop_on_axis_strehl =
        mean_from(closed_loop_values, first_steady_frame)
    mean_uncompensated_opd_rms_m =
        mean_from(uncompensated_opd_rms_values, first_steady_frame)
    mean_residual_opd_rms_m =
        mean_from(residual_opd_rms_values, first_steady_frame)
    mean_pdm_command_rms_m =
        mean_from(pdm_command_rms_values, first_steady_frame)
    return (;
        mean_open_loop_on_axis_strehl,
        mean_closed_loop_on_axis_strehl,
        improvement=mean_closed_loop_on_axis_strehl /
            mean_open_loop_on_axis_strehl,
        mean_uncompensated_opd_rms_m,
        mean_residual_opd_rms_m,
        mean_pdm_command_rms_m,
    )
end

function run_revolt_copper_validation()
    definition = process_definition()
    calibration = prepare_calibration_system(definition)
    streams = create_process_streams(definition)
    worker = nothing
    return mktempdir() do temporary_directory
        try
            worker = start_worker(definition, temporary_directory)
            signal = zeros(Float32, definition.signal_shape)
            flat_signal, interaction_matrix = calibrate_interaction_matrix!(
                worker,
                streams,
                calibration,
                signal;
                poke=definition.poke,
                report_progress=true,
            )
            norm(flat_signal) <= 1.0f-5 || error(
                "REVOLT Copper flat reference left a nonzero residual: " *
                "$(norm(flat_signal))",
            )
            all(isfinite, interaction_matrix) || error(
                "REVOLT Copper interaction matrix contains a non-finite value",
            )

            singular_values = svdvals(interaction_matrix)
            maximum_singular_value = maximum(singular_values)
            retained_tolerance =
                maximum_singular_value * definition.control_rcond
            retained_interaction_rank =
                count(>(retained_tolerance), singular_values)
            minimum_retained_rank = (4 * definition.command_count) ÷ 5
            retained_interaction_rank >= minimum_retained_rank || error(
                "REVOLT Copper interaction matrix retains only " *
                "$retained_interaction_rank control directions at rcond=" *
                "$(definition.control_rcond); expected at least " *
                "$minimum_retained_rank",
            )
            retained_interaction_condition =
                maximum_singular_value /
                singular_values[retained_interaction_rank]

            configure_worker_loop!(
                worker,
                streams,
                interaction_matrix,
                definition.gain,
                definition.control_rcond,
                temporary_directory,
            )
            atmosphere = close_atmospheric_loop!(
                worker,
                streams,
                definition,
                signal,
            )
            atmosphere.mean_closed_loop_on_axis_strehl > 0.35 || error(
                "REVOLT Copper did not produce a usable corrected on-axis " *
                "PSF: mean Strehl = " *
                "$(atmosphere.mean_closed_loop_on_axis_strehl)",
            )
            atmosphere.improvement >
                REVOLT_COPPER_MINIMUM_STREHL_IMPROVEMENT || error(
                "REVOLT Copper did not improve mean on-axis Strehl " *
                "sufficiently: ratio = $(atmosphere.improvement)",
            )
            atmosphere.mean_residual_opd_rms_m <
                0.4 * atmosphere.mean_uncompensated_opd_rms_m || error(
                "REVOLT Copper residual pupil OPD RMS was not reduced " *
                "sufficiently",
            )
            stop_worker!(worker)

            return (;
                system=definition.name,
                frame_shape=definition.frame_shape,
                signal_length=length(signal),
                command_count=definition.command_count,
                retained_interaction_rank,
                retained_interaction_condition,
                numerical_interaction_condition=
                    maximum_singular_value / minimum(singular_values),
                atmosphere...,
            )
        finally
            !isnothing(worker) && stop_worker_noexcept!(worker)
            close_process_streams!(streams)
        end
    end
end

function revolt_copper_main()
    result = run_revolt_copper_validation()
    println("REVOLT Copper/native-SHM pyRTC process loop passed")
    println("  detector frame shape: ", result.frame_shape)
    println("  signal length: ", result.signal_length)
    println("  PDM command length: ", result.command_count)
    println(
        "  retained interaction rank: ",
        result.retained_interaction_rank,
    )
    println(
        "  retained interaction condition: ",
        result.retained_interaction_condition,
    )
    println(
        "  numerical interaction condition: ",
        result.numerical_interaction_condition,
    )
    println(
        "  atmospheric mean open-loop on-axis Strehl: ",
        result.mean_open_loop_on_axis_strehl,
    )
    println(
        "  atmospheric mean closed-loop on-axis Strehl: ",
        result.mean_closed_loop_on_axis_strehl,
    )
    println("  atmospheric on-axis Strehl improvement: ", result.improvement)
    println(
        "  atmospheric mean OPD RMS: ",
        1.0e9 * result.mean_uncompensated_opd_rms_m,
        " -> ",
        1.0e9 * result.mean_residual_opd_rms_m,
        " nm",
    )
    println(
        "  mean PDM surface-OPD command RMS: ",
        1.0e9 * result.mean_pdm_command_rms_m,
        " nm",
    )
    return nothing
end

end # module PyRTCProcessHIL
