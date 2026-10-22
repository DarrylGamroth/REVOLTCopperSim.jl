# Isolated CUDA validation; no transport, external RTC, or installed services.
using AdaptiveOpticsSim, AdaptiveOpticsSim.AlgorithmGraphs
using REVOLTCopperSim, Dates, LinearAlgebra, SHA, Statistics, TOML
import CUDA

const AG = AdaptiveOpticsSim.AlgorithmGraphs
const OPTICS = AdaptiveOpticsSim.Optics
const EXPECTED = ((:default, 480, 32), (:fast, 240, 96), (:accurate, 1440, 256))

function cycle!(boundary)
    sequence = step_hil_frame!(boundary)
    adopt_hil_command!(boundary, sequence)
    return nothing
end

function checked_photons(system)
    photons = Array(graph_output(system.graph, :pwfs_photon_rate))
    @assert all(isfinite, photons) && all(>=(0), photons) && sum(photons) > 0
    @assert all(isfinite, hil_frame_buffer(system.boundary))
    return photons
end

function validate_inputs!(system, calibration, n)
    boundary = system.boundary
    fill!(hil_command_buffer(boundary), 0f0)
    cycle!(boundary)
    flat = checked_photons(system)
    if calibration
        axis = Float32.((2 .* (1:n) .- n .- 1) ./ n)
        probe = [1f-7 * (x + 0.3f0 * sinpi(3y)) for x in axis, y in axis]
        copyto!(system.uncompensated_opd, probe)
        CUDA.synchronize()
        cycle!(boundary)
        changed = checked_photons(system)
        @assert changed != flat "captured calibration must consume changed OPD"
        fill!(system.uncompensated_opd, 0f0)
        CUDA.synchronize()
        cycle!(boundary)
        @assert checked_photons(system) == flat "flat OPD must replay exactly"
        return Dict("nonuniform_input_changes_photons" => true, "flat_replay_exact" => true)
    end
    reset_hil_boundary!(boundary)
    first = Matrix{Float32}[]
    first_opd = Matrix{Float32}[]
    for _ in 1:2
        cycle!(boundary)
        push!(first, copy(hil_frame_buffer(boundary)))
        push!(first_opd, Array(graph_output(system.graph, :pupil_opd)))
        checked_photons(system)
    end
    @assert first_opd[1] != first_opd[2] "atmosphere must evolve"
    reset_hil_boundary!(boundary)
    for i in 1:2
        cycle!(boundary)
        @assert hil_frame_buffer(boundary) == first[i]
        @assert Array(graph_output(system.graph, :pupil_opd)) == first_opd[i]
        checked_photons(system)
    end
    return Dict("seeded_two_frame_replay_exact" => true, "atmosphere_evolves" => true)
end

function measurements(boundary)
    for _ in 1:32
        cycle!(boundary)
    end
    allocations = [@allocated(cycle!(boundary)) for _ in 1:16]
    @assert all(iszero, allocations) "captured HIL service must allocate zero host bytes"
    samples = zeros(UInt64, 128)
    for i in eachindex(samples)
        start = time_ns()
        cycle!(boundary)
        samples[i] = time_ns() - start
    end
    return Dict("samples_ns" => samples, "p50_ns" => median(samples),
        "p90_ns" => quantile(samples, 0.9),
        "minimum_ns" => minimum(samples), "maximum_ns" => maximum(samples),
        "mean_ns" => mean(samples), "host_bytes" => allocations)
end

function run_builder(mode, n, points, calibration, target)
    builder = calibration ? prepare_calibration_system : prepare_hil_system
    start = time_ns()
    system = builder(; mode, target, execution=CapturedGraphExecution())
    CUDA.synchronize()
    preparation_ns = time_ns() - start
    @assert (system.mode, system.pupil_resolution, system.modulation_points) == (mode, n, points)
    @assert size(hil_command_buffer(system.boundary)) == (277,)
    @assert size(hil_frame_buffer(system.boundary)) == (64, 64)
    pupil = calibration ? system.uncompensated_opd : graph_output(system.graph, :pupil_opd)
    @assert size(pupil) == (n, n)
    # Benchmark-only owner introspection verifies actual compact device ownership.
    prepared = AG.prepared_graph_node(system.graph, :pwfs).prepared
    modulation = prepared.front_end.modulation
    @assert OPTICS.modulation_point_count(modulation) == points
    @assert modulation.coordinates isa CUDA.CuArray
    @assert modulation.offsets isa CUDA.CuArray
    @assert size(modulation.coordinates) == (n,)
    @assert size(modulation.offsets) == (2, points)
    bytes = sizeof(eltype(modulation.coordinates)) * length(modulation.coordinates) +
        sizeof(eltype(modulation.offsets)) * length(modulation.offsets)
    checks = validate_inputs!(system, calibration, n)
    reset_hil_boundary!(system.boundary)
    timing = measurements(system.boundary)
    checked_photons(system)
    return Dict("mode" => string(mode), "builder" => calibration ? "calibration" : "production",
        "pupil_resolution" => n, "modulation_points" => points,
        "preparation_ns" => preparation_ns, "direct_modulation_gpu_bytes" => bytes,
        "coordinate_shape" => collect(size(modulation.coordinates)),
        "offset_shape" => collect(size(modulation.offsets)), "checks" => checks,
        "completed_hil_service" => timing)
end

function source_metadata(mod)
    root = dirname(dirname(pathof(mod)))
    return Dict("path" => root, "version" => string(Base.pkgversion(mod)),
        "revision" => readchomp(`git -C $root rev-parse HEAD`),
        "status" => readchomp(`git -C $root status --short`))
end

function persist(path, record)
    temporary = path * ".partial"
    open(temporary, "w") do io
        TOML.print(io, record; sorted=true)
    end
    mv(temporary, path; force=true)
end

function optional_text(path)
    return isfile(path) ? strip(read(path, String)) : "unavailable"
end

function main()
    length(ARGS) == 1 || error("usage: copper_modes_cuda.jl OUTPUT.toml")
    Threads.nthreads() == 1 || error("run with one Julia thread")
    BLAS.set_num_threads(1)
    AdaptiveOpticsSim.Backends.set_fft_provider_threads!(1)
    CUDA.allowscalar(false)
    target = AdaptiveOpticsSim.Backends.compute_device(CUDA.zeros(Float32, 1))
    output = abspath(ARGS[1])
    mkpath(dirname(output))
    record = Dict{String,Any}("started_utc" => string(now(UTC)),
        "julia_version" => string(VERSION), "cuda_version" => string(Base.pkgversion(CUDA)),
        "device" => CUDA.name(CUDA.device()), "julia_threads" => Threads.nthreads(),
        "cuda_driver_version" => string(CUDA.driver_version()),
        "cuda_runtime_version" => string(CUDA.runtime_version()),
        "machine" => Sys.MACHINE, "kernel" => string(Sys.KERNEL),
        "cpu_model" => first(Sys.cpu_info()).model,
        "process_status" => optional_text("/proc/self/status"),
        "load_average_at_start" => optional_text("/proc/loadavg"),
        "cpu0_governor" => optional_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "active_project" => something(Base.active_project(), ""),
        "harness_sha256" => bytes2hex(sha256(read(@__FILE__))),
        "copper" => source_metadata(REVOLTCopperSim), "aos" => source_metadata(AdaptiveOpticsSim),
        "protocol" => "Public captured constructors; flat/nonuniform calibration, two-frame seeded production replay; 32 warmups, 16 allocation checks, 128 completed host-visible HIL service samples. Zero commands, no RTC/transport. One sequential run per case, self-paced service distributions with no arrival process; no independent-run dispersion or tail-SLO claim. Numerical selection and replay only; not a physical accuracy or deadline qualification.",
        "results" => Dict{String,Any}[], "complete" => false)
    for filename in ("Project.toml", "Manifest.toml")
        path = joinpath(dirname(Base.active_project()), filename)
        isfile(path) && (record[lowercase(filename) * "_sha256"] = bytes2hex(sha256(read(path))))
    end
    persist(output, record)
    for (mode, n, points) in EXPECTED, calibration in (true, false)
        println("Preparing ", mode, " ", calibration ? "calibration" : "production"); flush(stdout)
        try
            push!(record["results"], run_builder(mode, n, points, calibration, target))
            persist(output, record)
        catch err
            record["failure"] = sprint(showerror, err)
            persist(output, record)
            rethrow()
        finally
            GC.gc(true)
            CUDA.reclaim()
        end
    end
    record["complete"] = true
    record["finished_utc"] = string(now(UTC))
    persist(output, record)
end

main()
