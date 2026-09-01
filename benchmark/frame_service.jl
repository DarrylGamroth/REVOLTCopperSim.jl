using AdaptiveOpticsSim
using AdaptiveOpticsSim.AlgorithmGraphs
using Dates
using LinearAlgebra
using Statistics
using TOML

const PACKAGE_ROOT = dirname(@__DIR__)
const AOS_ROOT = normpath(joinpath(PACKAGE_ROOT, "..", "AdaptiveOpticsSim.jl"))
const PACKAGE_DIRECTORY_NAME = basename(PACKAGE_ROOT)

if PACKAGE_DIRECTORY_NAME == "REVOLTClassicSim.jl"
    @eval import REVOLTClassicSim
    const InstrumentPackage = REVOLTClassicSim
    const INSTRUMENT_NAME = "REVOLT Classic"
    const ARTIFACT_PREFIX = "REVOLT-CLASSIC"
    const FRAME_SHAPE = (352, 352)
    const SENSOR = "diffractive Shack-Hartmann plus C-BLUE One IMX425"
    const PUPIL_RESOLUTION = 240
elseif PACKAGE_DIRECTORY_NAME == "REVOLTCopperSim.jl"
    @eval import REVOLTCopperSim
    const InstrumentPackage = REVOLTCopperSim
    const INSTRUMENT_NAME = "REVOLT Copper"
    const ARTIFACT_PREFIX = "REVOLT-COPPER"
    const FRAME_SHAPE = (64, 64)
    const SENSOR = "32-point shifted-mask Pyramid plus EMCCD"
    const PUPIL_RESOLUTION = 480
else
    error("unsupported instrument package directory '$PACKAGE_DIRECTORY_NAME'")
end

const BACKEND_NAME = lowercase(get(
    ENV,
    "REVOLT_BENCH_BACKEND",
    isempty(ARGS) ? "cpu" : first(ARGS),
))
const EXECUTION_NAME = lowercase(get(
    ENV,
    "REVOLT_BENCH_EXECUTION",
    BACKEND_NAME == "cpu" ? "stream" : "captured",
))
const SAMPLE_COUNT = parse(Int, get(ENV, "REVOLT_BENCH_SAMPLES", "20"))
const DEFAULT_WARMUP_CYCLES = BACKEND_NAME == "cpu" ? 10 : 200
const WARMUP_CYCLES = parse(Int, get(
    ENV,
    "REVOLT_BENCH_WARMUP",
    string(DEFAULT_WARMUP_CYCLES),
))
const RUN_COUNT = parse(Int, get(ENV, "REVOLT_BENCH_RUNS", "3"))
const OUTPUT_PATH = get(ENV, "REVOLT_BENCH_OUTPUT", "")

if BACKEND_NAME == "amdgpu"
    @eval import AMDGPU
elseif BACKEND_NAME == "cuda"
    @eval import CUDA
elseif BACKEND_NAME != "cpu"
    error("unsupported backend '$BACKEND_NAME'; use cpu, amdgpu, or cuda")
end

if EXECUTION_NAME == "captured" && BACKEND_NAME == "cpu"
    error("captured graph execution requires an accelerator backend")
elseif EXECUTION_NAME != "stream" && EXECUTION_NAME != "captured"
    error("unsupported execution '$EXECUTION_NAME'; use stream or captured")
end

@inline graph_execution() = EXECUTION_NAME == "captured" ?
    CapturedGraphExecution() : StreamGraphExecution()

function command_output(command::Cmd)
    try
        return readchomp(command)
    catch
        return "unknown"
    end
end

function optional_file(path::AbstractString)
    try
        return strip(read(path, String))
    catch
        return "unknown"
    end
end

function allowed_cpu_list()
    status = optional_file("/proc/self/status")
    status == "unknown" && return status
    for line in eachline(IOBuffer(status))
        startswith(line, "Cpus_allowed_list:") || continue
        return strip(last(split(line, ':'; limit=2)))
    end
    return "unknown"
end

function configure_benchmark!()
    Threads.nthreads() == 1 || error(
        "REVOLT frame-service evidence requires one Julia thread",
    )
    SAMPLE_COUNT > 0 || error("REVOLT_BENCH_SAMPLES must be positive")
    WARMUP_CYCLES >= 0 || error("REVOLT_BENCH_WARMUP must be nonnegative")
    RUN_COUNT > 0 || error("REVOLT_BENCH_RUNS must be positive")
    BLAS.set_num_threads(1)
    AdaptiveOpticsSim.Backends.set_fft_provider_threads!(1)
    return nothing
end

function target_and_accelerator_info()
    if BACKEND_NAME == "cpu"
        return (
            AdaptiveOpticsSim.Backends.HostComputeDevice(),
            Dict{String,Any}("device" => "host CPU"),
        )
    elseif BACKEND_NAME == "amdgpu"
        probe = AMDGPU.ROCArray(zeros(Float32, 1))
        device = AMDGPU.device()
        return (
            AdaptiveOpticsSim.Backends.compute_device(probe),
            Dict{String,Any}(
                "package_version" => string(Base.pkgversion(AMDGPU)),
                "device" => string(AMDGPU.HIP.name(device)),
                "gcn_architecture" => string(AMDGPU.HIP.gcn_arch(device)),
                "wavefront_size" => Int(AMDGPU.HIP.wavefrontsize(device)),
                "hip_runtime_version" => string(AMDGPU.HIP.runtime_version()),
            ),
        )
    end

    probe = CUDA.CuArray(zeros(Float32, 1))
    device = CUDA.device()
    return (
        AdaptiveOpticsSim.Backends.compute_device(probe),
        Dict{String,Any}(
            "package_version" => string(Base.pkgversion(CUDA)),
            "device" => CUDA.name(device),
            "compute_capability" => string(CUDA.capability(device)),
            "runtime_version" => string(CUDA.runtime_version()),
            "driver_version" => string(CUDA.driver_version()),
            "compiler_version" => string(CUDA.compiler_version()),
        ),
    )
end

function git_environment(root::AbstractString)
    tracked_status = command_output(
        `git -C $root status --porcelain=v1 --untracked-files=no`,
    )
    untracked_paths = command_output(
        `git -C $root ls-files --others --exclude-standard`,
    )
    return Dict{String,Any}(
        "root" => root,
        "commit" => command_output(`git -C $root rev-parse HEAD`),
        "branch" => command_output(`git -C $root branch --show-current`),
        "tracked_dirty" =>
            !isempty(tracked_status) && tracked_status != "unknown",
        "tracked_status_porcelain" => tracked_status,
        "untracked_paths" => untracked_paths,
    )
end

function timer_overhead_ns(samples::Int=10_000)
    values = Vector{Int64}(undef, samples)
    for index in eachindex(values)
        start = time_ns()
        values[index] = Int64(time_ns() - start)
    end
    return minimum(values)
end

@inline function run_cycle!(boundary)
    sequence = step_hil_frame!(boundary)
    adopt_hil_command!(boundary, sequence)
    return sequence
end

function validate_frame(boundary)
    frame = hil_frame_buffer(boundary)
    size(frame) == FRAME_SHAPE || error(
        "unexpected $INSTRUMENT_NAME frame shape $(size(frame))",
    )
    all(isfinite, frame) || error(
        "$INSTRUMENT_NAME produced a non-finite detector frame",
    )
    frame_sum = sum(frame)
    isfinite(frame_sum) && frame_sum > 0 || error(
        "$INSTRUMENT_NAME produced an empty detector frame",
    )
    return Float64(frame_sum)
end

function nearest_rank(values::Vector{Int64}, percentile::Float64)
    sorted = sort(values)
    index = clamp(ceil(Int, percentile * length(sorted)), 1, length(sorted))
    return sorted[index]
end

function summarize(values::Vector{Int64})
    result = Dict{String,Any}(
        "count" => length(values),
        "mean_ns" => mean(values),
        "minimum_ns" => minimum(values),
        "p50_ns" => nearest_rank(values, 0.50),
        "p90_ns" => nearest_rank(values, 0.90),
        "maximum_ns" => maximum(values),
    )
    if length(values) >= 100
        result["p99_ns"] = nearest_rank(values, 0.99)
    end
    return result
end

function measure_run(target, run_index::Int)
    preparation_start = time_ns()
    system = InstrumentPackage.prepare_hil_system(
        target=target,
        profile=:grid_gaussian,
        execution=graph_execution(),
    )
    preparation_ns = Int64(time_ns() - preparation_start)
    boundary = system.boundary

    for _ in 1:WARMUP_CYCLES
        run_cycle!(boundary)
        validate_frame(boundary)
    end

    GC.gc()
    allocation_bytes = @allocated run_cycle!(boundary)
    validate_frame(boundary)

    frame_service_ns = Vector{Int64}(undef, SAMPLE_COUNT)
    command_adoption_ns = Vector{Int64}(undef, SAMPLE_COUNT)
    cycle_service_ns = Vector{Int64}(undef, SAMPLE_COUNT)
    final_frame_sum = 0.0
    for sample_index in 1:SAMPLE_COUNT
        cycle_start = time_ns()
        sequence = step_hil_frame!(boundary)
        frame_ready = time_ns()
        adopt_hil_command!(boundary, sequence)
        cycle_ready = time_ns()
        frame_service_ns[sample_index] = Int64(frame_ready - cycle_start)
        command_adoption_ns[sample_index] = Int64(cycle_ready - frame_ready)
        cycle_service_ns[sample_index] = Int64(cycle_ready - cycle_start)
        final_frame_sum = validate_frame(boundary)
    end

    return Dict{String,Any}(
        "run" => run_index,
        "preparation_ns" => preparation_ns,
        "warmed_cycle_allocation_bytes" => allocation_bytes,
        "final_sequence" => Int(graph_step_sequence(system.graph)),
        "final_frame_sum" => final_frame_sum,
        "frame_service" => summarize(frame_service_ns),
        "command_adoption" => summarize(command_adoption_ns),
        "cycle_service" => summarize(cycle_service_ns),
        "raw_frame_service_ns" => frame_service_ns,
        "raw_command_adoption_ns" => command_adoption_ns,
        "raw_cycle_service_ns" => cycle_service_ns,
    )
end

function aggregate_runs(runs::Vector{Dict{String,Any}}, field::String)
    raw_field = "raw_$(field)_ns"
    values = reduce(vcat, Vector{Int64}(run[raw_field]) for run in runs)
    summary = summarize(values)
    run_means = Float64[
        Float64(run[field]["mean_ns"])
        for run in runs
    ]
    summary["mean_of_run_means_ns"] = mean(run_means)
    summary["run_mean_minimum_ns"] = minimum(run_means)
    summary["run_mean_maximum_ns"] = maximum(run_means)
    summary["throughput_hz_from_mean"] = 1.0e9 / summary["mean_ns"]
    return summary
end

function environment_record(accelerator)
    cpu = first(Sys.cpu_info())
    load_1m, load_5m, load_15m = Sys.loadavg()
    return Dict{String,Any}(
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
        "host_name" => command_output(`hostname`),
        "kernel" => string(Sys.KERNEL),
        "kernel_release" => command_output(`uname -r`),
        "architecture" => string(Sys.ARCH),
        "cpu_model" => cpu.model,
        "logical_cpu_threads" => Sys.CPU_THREADS,
        "load_average_1m" => load_1m,
        "load_average_5m" => load_5m,
        "load_average_15m" => load_15m,
        "allowed_cpus" => allowed_cpu_list(),
        "scaling_governor_cpu0" => optional_file(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
        ),
        "julia_version" => string(VERSION),
        "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "fft_threads" => 1,
        "active_project" => something(Base.active_project(), "unknown"),
        "adaptive_optics_sim_version" => string(
            Base.pkgversion(AdaptiveOpticsSim),
        ),
        "instrument_package_version" => string(
            Base.pkgversion(InstrumentPackage),
        ),
        "backend" => BACKEND_NAME,
        "accelerator" => accelerator,
        "package_source" => git_environment(PACKAGE_ROOT),
        "adaptive_optics_sim_source" => git_environment(AOS_ROOT),
    )
end

function benchmark_evidence()
    configure_benchmark!()
    target, accelerator = target_and_accelerator_info()
    overhead_ns = timer_overhead_ns()
    runs = Dict{String,Any}[
        measure_run(target, run_index)
        for run_index in 1:RUN_COUNT
    ]
    return Dict{String,Any}(
        "artifact_id" => "$(ARTIFACT_PREFIX)-FRAME-SERVICE-$(uppercase(BACKEND_NAME))-$(uppercase(EXECUTION_NAME))",
        "evidence_class" => "repeated warmed synchronized diagnostic profile",
        "contract" => Dict{String,Any}(
            "load_model" => "serial closed loop with immediate zero-command response",
            "arrival_model" => "the next frame starts only after the preceding command is adopted",
            "coordinated_omission_correction" => false,
            "execution_policy" => EXECUTION_NAME == "captured" ?
                "CapturedGraphExecution" : "StreamGraphExecution",
            "graph_profile" => "grid_gaussian",
            "frame_service_boundary" => "complete graph step through host-visible detector frame",
            "command_adoption_boundary" => "finite host command validation through target-ready command copy",
            "cycle_service_boundary" => "frame service plus immediate command adoption; external RTC work excluded",
            "tail_claim" => "diagnostic only; no fixed-arrival or production tail-latency claim",
            "timer" => "time_ns; overhead recorded but not subtracted",
            "timer_minimum_overhead_ns" => overhead_ns,
            "samples_per_run" => SAMPLE_COUNT,
            "warmup_cycles_per_run" => WARMUP_CYCLES,
            "runs_per_process" => RUN_COUNT,
        ),
        "workload" => Dict{String,Any}(
            "instrument" => INSTRUMENT_NAME,
            "sensor" => SENSOR,
            "pupil_resolution" => PUPIL_RESOLUTION,
            "frame_shape" => collect(FRAME_SHAPE),
            "command_count" => InstrumentPackage.command_count(),
            "atmosphere" => "deterministic evolving five-layer profile",
            "detector_noise" => true,
        ),
        "environment" => environment_record(accelerator),
        "aggregate" => Dict{String,Any}(
            "frame_service" => aggregate_runs(runs, "frame_service"),
            "command_adoption" => aggregate_runs(runs, "command_adoption"),
            "cycle_service" => aggregate_runs(runs, "cycle_service"),
        ),
        "runs" => runs,
    )
end

evidence = benchmark_evidence()
if isempty(OUTPUT_PATH)
    TOML.print(stdout, evidence; sorted=true)
else
    mkpath(dirname(abspath(OUTPUT_PATH)))
    open(OUTPUT_PATH, "w") do io
        TOML.print(io, evidence; sorted=true)
    end
    println(abspath(OUTPUT_PATH))
end
