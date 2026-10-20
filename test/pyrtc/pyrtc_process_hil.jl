module PyRTCProcessHIL

using AdaptiveOpticsSim.AlgorithmGraphs
import AdaptiveOpticsCalibration
using LinearAlgebra
import REVOLTCopperSim

include(joinpath(@__DIR__, "..", "science_analysis.jl"))

include(joinpath(@__DIR__, "pyrtc_shared_memory.jl"))
using .PyRTCSharedMemory

const PYRTC_STREAM_NAMES = ("wfs", "wfc", "signal", "signal2D")
const WORKER_PREFIX = "REVOLT_COPPER_PYRTC_WORKER "
const ATMOSPHERE_VALIDATION_BURN_IN_FRAMES = 10
# SplitMix64 seed 1 produces approximately 21× mean Strehl improvement for
# this deterministic fixture. Fifteen leaves a material regression margin.
const REVOLT_COPPER_MINIMUM_STREHL_IMPROVEMENT = 15.0
const AOC = AdaptiveOpticsCalibration
const InteractionMatrices = AOC.InteractionMatrices
const Reconstructors = AOC.Reconstructors
const AOC_INTERACTION_MATRIX_RELATIVE_TOLERANCE = 5.0f-6
const AOC_RECONSTRUCTOR_RELATIVE_TOLERANCE = 2.0f-4

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
@inline prepare_science_analysis(prepared) =
    prepare_science_analysis_fixture(
        prepared.graph, Val(:pwfs); wavelength_m=550.0e-9)

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
    responses = Matrix{Float32}(
        undef,
        2 * length(hil_command_buffer(boundary)),
        length(signal),
    )

    for command_index in eachindex(hil_command_buffer(boundary))
        if report_progress && (command_index == 1 ||
                command_index % 25 == 0 ||
                command_index == lastindex(hil_command_buffer(boundary)))
            println(
                "  calibrated command ",
                command_index,
                " / ",
                length(hil_command_buffer(boundary)),
            )
        end
        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        process_frame!(
            worker,
            streams,
            hil_frame_buffer(boundary),
            signal,
        )
        copyto!(@view(responses[2 * command_index - 1, :]), signal)

        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[command_index] = -poke
        adopt_hil_command!(boundary, sequence)
        sequence = step_hil_frame!(boundary)
        process_frame!(
            worker,
            streams,
            hil_frame_buffer(boundary),
            signal,
        )
        copyto!(@view(responses[2 * command_index, :]), signal)
    end

    fill!(hil_command_buffer(boundary), 0.0f0)
    adopt_hil_command!(boundary, sequence)
    reset_hil_boundary!(boundary)
    return flat_signal, responses
end

"""Return the direct central-difference matrix used by the pre-AOC Copper gate."""
function central_difference_interaction_matrix(
    responses::Matrix{Float32},
    poke::Float32,
)
    iseven(size(responses, 1)) || throw(DimensionMismatch(
        "push-pull responses must have an even row count",
    ))
    interaction_matrix = Matrix{Float32}(
        undef,
        size(responses, 2),
        div(size(responses, 1), 2),
    )
    scale = 2.0f0 * poke
    for command_index in axes(interaction_matrix, 2)
        @views @. interaction_matrix[:, command_index] =
            (responses[2 * command_index - 1, :] -
             responses[2 * command_index, :]) / scale
    end
    return interaction_matrix
end

"""Form Copper's interaction matrix from an interleaved +poke/-poke response cycle."""
function interaction_matrix_from_responses(
    responses::Matrix{Float32},
    poke::Float32,
)
    iseven(size(responses, 1)) || throw(DimensionMismatch(
        "push-pull responses must have an even row count",
    ))
    method = InteractionMatrices.ZonalPushPull(
        fill(poke, div(size(responses, 1), 2)),
    )
    plan = AOC.prepare(
        method,
        InteractionMatrices.ResponseSpecification(size(responses, 2)),
    )
    product = AOC.process(plan, responses)
    return InteractionMatrices.interaction_matrix(product)
end

@inline relative_error(reference, approximation) = iszero(norm(reference)) ?
    norm(approximation) : norm(reference - approximation) / norm(reference)

"""Prepare Copper's 5%-cutoff, identity-weighted, zero-Tikhonov AOC reconstructor."""
function prepare_aoc_reconstructor(
    interaction_matrix::Matrix{Float32},
    retained_rank::Int,
    control_rcond::Float32,
)
    measurement_count, command_count = size(interaction_matrix)
    plan = AOC.prepare(
        Reconstructors.StrokeWeightedTikhonov(
            retained_rank;
            tikhonov_scale=0.0f0,
            rtol=control_rcond,
        ),
        Reconstructors.ReconstructorSpecification(
            measurement_count,
            command_count,
            Float32,
        ),
    )
    return AOC.process(
        plan,
        Reconstructors.StrokeWeightedReconstructorInputs(
            interaction_matrix,
            Matrix{Float32}(I, measurement_count, measurement_count),
            Matrix{Float32}(I, command_count, command_count),
        ),
    )
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

function validate_control_matrix(
    control_matrix::AbstractMatrix{Float32},
    expected_shape::Tuple{Int,Int},
    gain::Float32,
)
    size(control_matrix) == expected_shape || throw(DimensionMismatch(
        "control matrix has size $(size(control_matrix)); expected $expected_shape " *
        "(corrector by measurement)",
    ))
    all(isfinite, control_matrix) || throw(ArgumentError(
        "control matrix contains a non-finite value",
    ))
    isfinite(gain) || throw(ArgumentError("pyRTC loop gain must be finite"))
    return nothing
end

function configure_worker_control_matrix!(
    worker::PyRTCWorker,
    streams::ProcessStreams,
    control_matrix::Matrix{Float32},
    gain::Float32,
    temporary_directory::AbstractString,
)
    expected_shape = (
        only(stream_shape(streams.wfc)),
        only(stream_shape(streams.signal)),
    )
    validate_control_matrix(control_matrix, expected_shape, gain)
    matrix_path = joinpath(temporary_directory, "control_matrix.f32")
    open(matrix_path, "w") do io
        write(io, control_matrix)
    end
    send_worker_command!(
        worker,
        "CONFIGURE_CONTROL_MATRIX $matrix_path $(Float64(gain))",
        "CONTROL_MATRIX_CONFIGURED";
        timeout=30.0,
    )
    send_worker_command!(worker, "FLATTEN", "FLATTENED")
    flat_command = zeros(Float32, size(control_matrix, 1))
    read_next!(flat_command, streams.wfc; timeout=5.0)
    all(iszero, flat_command) || error(
        "pyRTC worker flatten command is nonzero",
    )
    return nothing
end

"""Read pyRTC's current corrector-by-measurement control matrix in Julia layout."""
function read_worker_control_matrix(
    worker::PyRTCWorker,
    expected_shape::Tuple{Int,Int},
    temporary_directory::AbstractString,
)
    matrix_path = joinpath(temporary_directory, "worker_control_matrix.f32")
    send_worker_command!(
        worker,
        "WRITE_CONTROL_MATRIX $matrix_path",
        "CONTROL_MATRIX_WRITTEN";
        timeout=30.0,
    )
    expected_bytes = sizeof(Float32) * prod(expected_shape)
    filesize(matrix_path) == expected_bytes || error(
        "pyRTC worker control matrix has $(filesize(matrix_path)) bytes; " *
        "expected $expected_bytes",
    )
    control_matrix = Matrix{Float32}(undef, expected_shape)
    open(matrix_path, "r") do io
        read!(io, control_matrix)
    end
    all(isfinite, control_matrix) || error(
        "pyRTC worker control matrix contains a non-finite value",
    )
    return control_matrix
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
    science_fixture = prepare_science_analysis(prepared)
    boundary = prepared.boundary
    command = zeros(Float32, definition.command_count)
    open_loop_values = Vector{Float32}(undef, frames)
    closed_loop_values = Vector{Float32}(undef, frames)
    uncompensated_opd_rms_values = Vector{Float64}(undef, frames)
    residual_opd_rms_values = Vector{Float64}(undef, frames)
    pdm_command_rms_values = Vector{Float64}(undef, frames)
    support = science_fixture.support
    sequence = step_hil_frame!(boundary)

    for frame_index in 1:frames
        atmosphere_opd = graph_output(prepared.graph, Val(:atmosphere_opd))
        residual_opd = graph_output(prepared.graph, Val(:pupil_opd))
        strehl = update_science_analysis!(
            science_fixture,
            atmosphere_opd,
            residual_opd,
        )
        open_loop_values[frame_index] =
            strehl.open_loop
        closed_loop_values[frame_index] =
            strehl.corrected
        uncompensated_opd_rms_values[frame_index] =
            pupil_opd_rms(atmosphere_opd, support)
        residual_opd_rms_values[frame_index] =
            pupil_opd_rms(residual_opd, support)
        maximum(science_fixture.open_loop_psf) <= 1.001f0 || error(
            "open-loop PSF exceeds its exact diffraction-limited peak",
        )
        maximum(science_fixture.corrected_psf) <= 1.001f0 || error(
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
            flat_signal, responses = calibrate_interaction_matrix!(
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
            interaction_matrix = interaction_matrix_from_responses(
                responses,
                definition.poke,
            )
            direct_interaction_matrix = central_difference_interaction_matrix(
                responses,
                definition.poke,
            )
            interaction_matrix_relative_error = relative_error(
                direct_interaction_matrix,
                interaction_matrix,
            )
            interaction_matrix_relative_error <=
                AOC_INTERACTION_MATRIX_RELATIVE_TOLERANCE || error(
                "AOC interaction matrix differs from Copper central difference: " *
                "$interaction_matrix_relative_error",
            )
            all(isfinite, interaction_matrix) || error(
                "REVOLT Copper AOC interaction matrix contains a non-finite value",
            )

            singular_values = svdvals(direct_interaction_matrix)
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

            aoc_product = prepare_aoc_reconstructor(
                interaction_matrix,
                retained_interaction_rank,
                definition.control_rcond,
            )
            aoc_product.diagnostics.retained_rank == retained_interaction_rank || error(
                "AOC retained rank does not match Copper's 5% cutoff",
            )
            aoc_product.diagnostics.numerical_rank == retained_interaction_rank || error(
                "AOC numerical rank $(aoc_product.diagnostics.numerical_rank) " *
                "does not match Copper's 5% cutoff rank " *
                "$retained_interaction_rank",
            )
            control_matrix = Reconstructors.reconstructor(aoc_product)
            configure_worker_loop!(
                worker,
                streams,
                direct_interaction_matrix,
                definition.gain,
                definition.control_rcond,
                temporary_directory,
            )
            baseline_control_matrix = read_worker_control_matrix(
                worker,
                size(control_matrix),
                temporary_directory,
            )
            reconstructor_relative_error = relative_error(
                baseline_control_matrix,
                control_matrix,
            )
            reconstructor_relative_error <= AOC_RECONSTRUCTOR_RELATIVE_TOLERANCE ||
                error(
                    "AOC reconstructor differs from Copper's current " *
                    "5%-cutoff pseudoinverse: $reconstructor_relative_error",
                )

            configure_worker_control_matrix!(
                worker,
                streams,
                control_matrix,
                definition.gain,
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
                interaction_matrix_relative_error,
                reconstructor_relative_error,
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
        "  AOC/direct interaction-matrix relative error: ",
        result.interaction_matrix_relative_error,
    )
    println(
        "  AOC/current-pseudoinverse relative error: ",
        result.reconstructor_relative_error,
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
