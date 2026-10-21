# No transport/RTC. Scientific inputs and reductions are in Float64 on host;
# the maintained Copper graph runs its ordinary Float32 CUDA implementation.
module CopperConvergence
using AdaptiveOpticsSim, AdaptiveOpticsSim.AlgorithmGraphs
using REVOLTCopperSim, LinearAlgebra, Statistics, TOML, SHA
import CUDA
const AOS = AdaptiveOpticsSim
const POKE = 2e-8
const PROBES = ("flat", "piston", "tilt_x_20nm", "tilt_y_20nm", "focus_20nm",
    "astigmatism_20nm", "coma_20nm", "tilt_x_100nm", "focus_100nm",
    "fourier_4_20nm", "fourier_8_20nm", "fourier_12_20nm", "mixed_100nm")
# Identical detector support at every grid: geometric radius15 in a30x30 ROI.
const SUPPORT = [(i - 15.5)^2 + (j - 15.5)^2 <= 15^2 for i in 1:30, j in 1:30]
const OFFSETS = findall(SUPPORT)
const ORIGINS = ((1, 1), (33, 1), (33, 33), (1, 33)) # zero-based q1,q2,q3,q4

function opd_probe(n, name)
    axis = (2 .* collect(1:n) .- n .- 1) ./ n
    [Float32(probe_value(name, x, y)) for x in axis, y in axis]
end
function probe_value(name, x, y)
    name == "flat" && return 0.0
    name == "piston" && return 100e-9
    name == "tilt_x_20nm" && return 20e-9*x
    name == "tilt_y_20nm" && return 20e-9*y
    name == "focus_20nm" && return 20e-9*(2*(x*x+y*y)-1)
    name == "astigmatism_20nm" && return 20e-9*(x*x-y*y)
    name == "coma_20nm" && return 20e-9*(3*(x*x+y*y)-2)*x
    name == "tilt_x_100nm" && return 100e-9*x
    name == "focus_100nm" && return 100e-9*(2*(x*x+y*y)-1)
    name == "fourier_4_20nm" && return 20e-9*sinpi(4*x)*cospi(3*y)
    name == "fourier_8_20nm" && return 20e-9*sinpi(8*x)*cospi(5*y)
    name == "fourier_12_20nm" && return 20e-9*sinpi(12*x)*cospi(7*y)
    name == "mixed_100nm" && return 100e-9*(0.5sinpi(x+2y)+0.3cospi(3x-y)+0.2sinpi(7x+5y))
    error("unknown probe $name")
end

"""I4Q differential, q1TL q2BL q3BR q4TR, fixed finest-flat incidence divisor."""
function raw_slopes(rate, divisor)
    result = Matrix{Float64}(undef, length(OFFSETS), 2)
    for (k, I) in enumerate(OFFSETS)
        i, j = Tuple(I)
        a, b, c, d = (Float64(rate[r+i, col+j]) for (r, col) in ORIGINS)
        result[k, 1] = (a-b+d-c)/divisor
        result[k, 2] = (a-d+b-c)/divisor
    end
    return vec(result)
end
function flat_divisor(rate)
    sum(sum(Float64(rate[r+I[1], c+I[2]]) for (r,c) in ORIGINS) for I in OFFSETS) / length(OFFSETS)
end

function prepare_case(n, points; calibration=true, target)
    definition = REVOLTCopperSim._graph_definition(:grid_gaussian, target;
        calibration, pupil_resolution=n, modulation_points=points)
    graph = prepare_algorithm_graph(definition; target, execution=CapturedGraphExecution())
    boundary = prepare_graph_hil_boundary(graph;
        command_input=:pdm_command, frame_output=:pwfs_frame)
    return (; graph, boundary)
end
function probe_rates(system, n)
    input = graph_input(system.graph, :uncompensated_opd)
    fill!(graph_input(system.graph, :pdm_command), 0f0)
    rates = Array{Float32}(undef, 64, 64, length(PROBES))
    for (index, name) in enumerate(PROBES)
        copyto!(input, opd_probe(n, name))
        CUDA.synchronize()
        step_graph!(system.graph)
        rates[:,:,index] .= Array(graph_output(system.graph, :pwfs_photon_rate))
    end
    fill!(input, 0f0)
    CUDA.synchronize()
    @assert all(isfinite, rates) && minimum(rates) >= 0
    return rates
end
function calibration_response(system, divisor; poke=POKE, columns=1:277)
    command = graph_input(system.graph, :pdm_command)
    host_command = zeros(Float32, 277)
    matrix = Matrix{Float32}(undef, 2length(OFFSETS), length(columns))
    for (col, actuator) in enumerate(columns)
        host_command[actuator] = Float32(poke)
        copyto!(command, host_command); CUDA.synchronize()
        step_graph!(system.graph)
        positive = raw_slopes(Array(graph_output(system.graph, :pwfs_photon_rate)), divisor)
        host_command[actuator] = -Float32(poke)
        copyto!(command, host_command); CUDA.synchronize()
        step_graph!(system.graph)
        negative = raw_slopes(Array(graph_output(system.graph, :pwfs_photon_rate)), divisor)
        matrix[:,col] .= (positive .- negative) ./ (2poke)
        host_command[actuator] = 0f0
        if col % 32 == 0
            println("calibration ", col, "/", length(columns)); flush(stdout)
        end
    end
    fill!(command, 0f0); CUDA.synchronize()
    @assert all(isfinite, matrix)
    return matrix
end
function cycle!(boundary)
    seq = step_hil_frame!(boundary)
    adopt_hil_command!(boundary, seq)
    nothing
end
function timings(f, value; warmups=32, samples=128)
    for _ in 1:warmups
        f(value)
    end
    allocations = [@allocated(f(value)) for _ in 1:16]
    times = zeros(UInt64, samples)
    gc_before = Base.gc_num()
    for i in eachindex(times)
        start = time_ns(); f(value); times[i] = time_ns()-start
    end
    return Dict("samples_ns"=>times, "p50_ns"=>median(times),
        "p90_ns"=>quantile(times,0.9), "host_bytes"=>allocations,
        "gc_collections"=>Base.GC_Diff(Base.gc_num(), gc_before).pause)
end
function write_array(directory, name, array)
    # Portable little-endian Float32, Julia column-major order, shape in metadata.
    ENDIAN_BOM == 0x04030201 || error("writer requires little-endian host")
    path = joinpath(directory, name*".f32.gz")
    open(pipeline(`gzip -n -c`, stdout=path), "w") do io
        write(io, vec(Float32.(array)))
    end
    return Dict("file"=>basename(path), "shape"=>collect(size(array)),
        "sha256"=>bytes2hex(sha256(read(path))), "type"=>"little-endian Float32 column-major")
end
function read_array(directory, record)
    bytes = read(`gzip -dc $(joinpath(directory, record["file"]))`)
    reshape(copy(reinterpret(Float32, bytes)), Tuple(record["shape"]))
end
function persist(path, data)
    temp = path*".tmp"
    open(io->TOML.print(io,data;sorted=true), temp, "w")
    mv(temp,path;force=true)
end
function relerr(a,b)
    denominator = norm(Float64.(b))
    denominator == 0 && return norm(Float64.(a).-b) == 0 ? 0.0 : Inf
    norm(Float64.(a).-b)/denominator
end

function run_case(directory, n, points; reference=nothing, full_calibration=true, benchmark=true)
    path = joinpath(directory,"n$(n)-m$(points).toml")
    isfile(path) && return TOML.parsefile(path)
    println("preparing n=",n," modulation=",points); flush(stdout)
    target = AOS.Backends.compute_device(CUDA.zeros(Float32,1))
    system = prepare_case(n,points;target)
    rates = probe_rates(system,n)
    divisor = isnothing(reference) ? flat_divisor(rates[:,:,1]) : reference["divisor"]
    # Pilot mode deliberately omits full calibration and timing; kept separate dirs.
    columns = full_calibration ? (1:277) : [1,7,50,100,130,139,148,177,228,271,277]
    matrix = calibration_response(system,divisor;columns)
    result = Dict{String,Any}("resolution"=>n,"modulation_points"=>points,
        "fft_side"=>64div(n,30), "divisor"=>divisor,
        "calibration_poke_m"=>POKE, "calibration_columns"=>collect(columns),
        "support_pixels"=>length(OFFSETS), "probes"=>collect(PROBES),
        "rates"=>write_array(directory,"n$(n)-m$(points)-rates",rates),
        "interaction"=>write_array(directory,"n$(n)-m$(points)-interaction",matrix),
        "aos_version"=>string(Base.pkgversion(AOS)),"cuda_version"=>string(Base.pkgversion(CUDA)),
        "julia"=>string(VERSION),"device"=>CUDA.name(CUDA.device()))
    if benchmark
        fill!(graph_input(system.graph,:uncompensated_opd),0f0); CUDA.synchronize()
        result["calibration_graph"] = timings(step_graph!,system.graph)
        reset_hil_boundary!(system.boundary)
        result["calibration_hil"] = timings(cycle!,system.boundary)
    end
    persist(path,result)
    # Return only host arrays; release exact graph owners and their captured workspaces.
    system=nothing; GC.gc(true); CUDA.reclaim()
    if benchmark
        atmospheric=prepare_case(n,points;calibration=false,target)
        result["atmosphere_graph"]=timings(step_graph!,atmospheric.graph)
        reset_hil_boundary!(atmospheric.boundary)
        result["atmosphere_hil"]=timings(cycle!,atmospheric.boundary)
        persist(path,result)
        atmospheric=nothing; GC.gc(true); CUDA.reclaim()
    end
    println("completed n=",n," modulation=",points); flush(stdout)
    return result
end
function main(args)
    length(args)>=2 || error("usage: copper_convergence.jl OUTPUT pilot|sweep|case [resolution points]")
    directory=abspath(args[1]); mkpath(directory)
    CUDA.functional() || error("CUDA hardware required")
    CUDA.allowscalar(false); BLAS.set_num_threads(1); AOS.Backends.set_fft_provider_threads!(1)
    pilot=args[2]=="pilot"
    # Highest reference first: every response uses exactly its incidence divisor.
    ref=run_case(directory,1440,256;full_calibration=!pilot,benchmark=!pilot)
    if args[2]=="case"
        run_case(directory,parse(Int,args[3]),parse(Int,args[4]);reference=ref)
    elseif pilot
        for (n,m) in ((960,128),(960,256),(1440,128),(480,32),(240,32),(120,32))
            run_case(directory,n,m;reference=ref,full_calibration=false,benchmark=false)
        end
    else
        for (n,m) in ((960,256),(1440,128))
            run_case(directory,n,m;reference=ref)
        end
        for n in (120,180,240,360,480,720), m in (8,16,24,32,48,64)
            run_case(directory,n,m;reference=ref)
        end
    end
end
end # module
if abspath(PROGRAM_FILE)==@__FILE__
    CopperConvergence.main(ARGS)
end
