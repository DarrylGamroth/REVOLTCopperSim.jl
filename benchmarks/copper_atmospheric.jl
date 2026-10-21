# Paired atmospheric and causal closed-loop discretization validation.
# Input generation, diagnostics and host controller are outside plant timing.
module CopperAtmospheric
include("copper_convergence.jl")
using .CopperConvergence
using LinearAlgebra, Statistics, TOML, SHA
const CC=CopperConvergence
const ATM=CC.AOS.Atmospheres
const SELF_SHA=bytes2hex(sha256(read(@__FILE__)))
const MASTER=960
const DT=0.002
const WAVELENGTH=550e-9
const WEIGHTS=(.45,.1,.1,.25,.1)
const SPEED=(5.,5.,7.,10.,10.)
const ANGLES=(0.,72.,144.,216.,288.)
const CASES=((1440,256),(480,32),(180,64),(240,96),(240,128))

function science_fixture(n)
    axis=(2 .* collect(1:n).-n.-1)./n
    support=[x*x+y*y<=1 for x in axis,y in axis]
    basis=[exp(-(x-(i-10)*.125)^2/(2CC.REVOLTCopperSim._PDM_INFLUENCE_WIDTH^2)) for x in axis,i in 1:19]
    (;support,basis,indices=CC.REVOLTCopperSim.actuator_grid_indices(),grid=zeros(19,19))
end
function science(fixture,opd,command)
    fill!(fixture.grid,0)
    fixture.grid[fixture.indices].=command
    surface=fixture.basis*fixture.grid*fixture.basis'
    residual=Float64.(opd).+surface
    values=residual[fixture.support]
    rms=std(values;corrected=false)
    # Exact on-axis coherent peak / ideal peak for a uniform binary aperture.
    strehl=abs2(mean(cispi.(2 .* values ./ WAVELENGTH)))
    return rms,strehl
end
function initialized_screens(seed;master=MASTER)
    telescope=CC.AOS.Optics.Telescope(;resolution=master,diameter=1.22,T=Float32)
    atm=ATM.MultiLayerAtmosphere(telescope;r0=.15,reference_wavelength_m=500e-9,L0=30.,
        fractional_cn2=collect(WEIGHTS),wind_speed=collect(SPEED),
        wind_direction_deg=collect(ANGLES),altitude=[0.,1000.,5000.,10000.,12000.],T=Float32)
    ATM.initialize_atmosphere!(atm,CC.AOS.runtime_rng(seed))
    return [copy(layer.generator.state.phase_rad) for layer in atm.layers]
end
function render!(out,screens,frame,r0;master=MASTER)
    fill!(out,0)
    scratch=similar(out)
    for k in eachindex(screens)
        t=frame*DT
        dx=Float32(SPEED[k]*cosd(ANGLES[k])*t/(1.22/master))
        dy=Float32(SPEED[k]*sind(ANGLES[k])*t/(1.22/master))
        scale=Float32(sqrt(WEIGHTS[k])*500e-9/(2pi)*(.15/r0)^(5/6))
        ATM.extract_shifted_screen!(scratch,screens[k],dx,dy,scale,Float32(master/size(out,1)))
        out .+= scratch
    end
    return out
end
function calibration(n,m,directory,divisor)
    record=TOML.parsefile(joinpath(directory,"n$(n)-m$(m).toml"))
    @assert record["full_calibration"] && record["calibration_columns"]==collect(1:277)
    @assert record["divisor"]==divisor
    @assert record["graph_sha256"]==CC.GRAPH_SHA && record["aos_revision"]==CC.AOS_REVISION
    @assert record["interaction"]["sha256"]==bytes2hex(sha256(read(joinpath(directory,record["interaction"]["file"]))))
    a=CC.read_array(directory,record["interaction"])
    @assert size(a)==(1432,277)
    f=svd(Float64.(a);full=false)
    keep=f.S .> .05maximum(f.S)
    inverse=f.V[:,keep]*Diagonal(1 ./ f.S[keep])*f.U[:,keep]'
    return inverse,count(keep)
end
function expose(system,opd,command)
    copyto!(CC.graph_input(system.graph,:uncompensated_opd),opd)
    copyto!(CC.graph_input(system.graph,:pdm_command),command)
    CC.CUDA.synchronize()
    CC.step_graph!(system.graph)
    Array(CC.graph_output(system.graph,:pwfs_photon_rate))
end
function trajectory(system,screens,n,m,r0,seed,frames,calibration_dir,divisor,output;gain=.15,fine_system=nothing,fine_flat=nothing)
    name="n$(n)-m$(m)-r$(round(Int,100r0))-s$(seed)"
    path=joinpath(output,name*".toml")
    screen_hashes=[bytes2hex(sha256(reinterpret(UInt8,vec(Array(s))))) for s in screens]
    if isfile(path)
        previous=TOML.parsefile(path)
        @assert previous["harness_sha256"]==SELF_SHA && previous["graph_sha256"]==CC.GRAPH_SHA
        @assert previous["aos_revision"]==CC.AOS_REVISION && previous["screen_hashes"]==screen_hashes
        @assert previous["frames"]==frames && previous["gain"]==gain
        println("verified cached ",name);flush(stdout)
        return
    end
    calibration_n,calibration_m=(n,m) in ((1920,256),(1440,512)) ? (1440,256) : (n,m)
    inverse,rank=calibration(calibration_n,calibration_m,calibration_dir,divisor)
    command=zeros(Float32,277)
    opd=CC.CUDA.zeros(Float32,n,n)
    science_opd=CC.CUDA.zeros(Float32,480,480)
    fixture=science_fixture(480)
    flat=expose(system,opd,command)
    flat_signal=CC.raw_slopes(flat,divisor)
    open_frame_ids=sort!(unique!([1;collect(10:10:frames);frames]))
    rates=Array{Float32}(undef,64,64,length(open_frame_ids))
    signals=Matrix{Float32}(undef,1432,length(open_frame_ids))
    closed_signals=Matrix{Float32}(undef,1432,frames)
    commands=Matrix{Float32}(undef,277,frames)
    diagnostics=zeros(Float64,5,frames)
    zero_command=zeros(Float32,277)
    checks=Dict{String,Any}[]
    fine_opd=CC.CUDA.zeros(Float32,1440,1440)
    for frame in 1:frames
        render!(opd,screens,frame,r0)
        open_index=findfirst(==(frame),open_frame_ids)
        if !isnothing(open_index)
            open=expose(system,opd,zero_command)
            rates[:,:,open_index].=open
            signals[:,open_index].=CC.raw_slopes(open,divisor).-flat_signal
        end
        closed=expose(system,opd,command)
        frame==1 && @assert CC.relerr(closed,open)<1e-6 "zero-command first frame must equal open loop"
        signal=CC.raw_slopes(closed,divisor).-flat_signal
        closed_signals[:,frame].=signal
        commands[:,frame].=command # command actually applied to this frame
        render!(science_opd,screens,frame,r0)
        host_opd=Array(science_opd)
        residual,strehl=science(fixture,host_opd,command)
        open_rms,open_strehl=science(fixture,host_opd,zero_command)
        diagnostics[:,frame].=(residual,strehl,open_rms,open_strehl,sqrt(mean(abs2,command)))
        if frame==1 || frame%50==0 || frame==frames
            check=Dict{String,Any}("frame"=>frame)
            # Independent common science-grid convergence, holding command fixed.
            for dense in (960,1440)
                dense_opd=CC.CUDA.zeros(Float32,dense,dense)
                render!(dense_opd,screens,frame,r0)
                dense_rms,dense_strehl=science(science_fixture(dense),Array(dense_opd),command)
                check["science_$(dense)_rms_m"]=dense_rms
                check["science_$(dense)_strehl"]=dense_strehl
            end
            if !isnothing(fine_system) && (n,m)!=(1440,256)
                render!(fine_opd,screens,frame,r0)
                fine_rate=expose(fine_system,fine_opd,command)
                fine_signal=CC.raw_slopes(fine_rate,divisor).-CC.raw_slopes(fine_flat,divisor)
                check["same_command_rate_relative_l2"]=CC.relerr(closed,fine_rate)
                check["same_command_slope_relative_l2"]=CC.relerr(signal,fine_signal)
                check["same_command_slope_error_rms"]=sqrt(mean(abs2,signal.-fine_signal))
                check["fine_rate"]=CC.write_array(output,"n$(n)-m$(m)-r$(round(Int,100r0))-s$(seed)-f$(frame)-fine-rate",fine_rate)
                check["own_rate"]=CC.write_array(output,"n$(n)-m$(m)-r$(round(Int,100r0))-s$(seed)-f$(frame)-own-rate",closed)
            end
            push!(checks,check)
        end
        command .-= Float32.(gain .* (inverse*signal))
        @assert all(isfinite,command) && maximum(abs,command)<1e-3 "unstable control"
        frame%50==0 && (println("n=$n m=$m r0=$r0 seed=$seed frame=$frame");flush(stdout))
    end
    @assert maximum(abs,commands[:,2:end].-commands[:,1:end-1].+gain.*(inverse*closed_signals[:,1:end-1]))<1e-11
    data=Dict("resolution"=>n,"modulation_points"=>m,"r0_m"=>r0,"seed"=>seed,
        "open_frame_ids"=>open_frame_ids,"science_resolution"=>480,"science_wavelength_m"=>WAVELENGTH,"frames"=>frames,"burn_in"=>min(50,frames÷3),"dt_s"=>DT,"gain"=>gain,"rcond"=>.05,
        "calibration_resolution"=>calibration_n,"calibration_points"=>calibration_m,"checks"=>checks,"retained_rank"=>rank,"divisor"=>divisor,"master_pupil_resolution"=>MASTER,
        "harness_sha256"=>SELF_SHA,"base_harness_sha256"=>CC.HARNESS_SHA,
        "graph_sha256"=>CC.GRAPH_SHA,"aos_revision"=>CC.AOS_REVISION,"copper_revision"=>CC.COPPER_REVISION,
        "screen_hashes"=>screen_hashes,
        "rates"=>CC.write_array(output,name*"-rates",rates),
        "open_slopes"=>CC.write_array(output,name*"-open-slopes",signals),
        "closed_slopes"=>CC.write_array(output,name*"-closed-slopes",closed_signals),
        "commands"=>CC.write_array(output,name*"-commands",commands),
        "diagnostic_rows"=>["residual_opd_rms_m","on_axis_strehl","open_opd_rms_m","open_on_axis_strehl","command_rms_m"],
        "diagnostics"=>CC.write_array(output,name*"-diagnostics",diagnostics),
        "flat"=>CC.write_array(output,name*"-flat",flat))
    # Static warmed plant-only timing. Controller, uploads, atmosphere sampling and
    # diagnostics intentionally excluded; previous end-to-end HIL benchmark retained.
    data["plant_timing"]=(n,m) in CASES[2:end] ? CC.timings(CC.step_graph!,system.graph;samples=128) :
        CC.timings(CC.step_graph!,system.graph;warmups=4,samples=16)
    CC.persist(joinpath(output,name*".toml"),data)
    println("completed ",name);flush(stdout)
end
function main(args)
    length(args)>=2 || error("usage: OUTPUT CALIBRATION_DIR [frames=300] [seedlist=1,2,3] [r0list=.10,.15,.20] [pairs]")
    output=abspath(args[1]);mkpath(output)
    calibration_dir=abspath(args[2])
    frames=length(args)>=3 ? parse(Int,args[3]) : 300
    seeds=length(args)>=4 ? parse.(Int,split(args[4],',')) : [1,2,3]
    strengths=length(args)>=5 ? parse.(Float64,split(args[5],',')) : [.10,.15,.20]
    cases=length(args)>=6 ? [Tuple(parse.(Int,split(p,':'))) for p in split(args[6],',')] : CASES
    CC.CUDA.functional() || error("CUDA hardware required")
    CC.CUDA.allowscalar(false);BLAS.set_num_threads(1);CC.AOS.Backends.set_fft_provider_threads!(1)
    target=CC.AOS.Backends.compute_device(CC.CUDA.zeros(Float32,1))
    ref=TOML.parsefile(joinpath(calibration_dir,"n1440-m256.toml"));divisor=ref["divisor"]
    fine_flat=CC.read_array(calibration_dir,ref["rates"])[:,:,1]
    for seed in seeds
        println("initializing shared native atmosphere seed=",seed);flush(stdout)
        host=initialized_screens(seed)
        screens=CC.CUDA.CuArray.(host);host=nothing;GC.gc(true)
        fine_system=any(pair->pair in CASES,cases) ? CC.prepare_case(1440,256;target) : nothing
        for (n,m) in cases
            system=(n,m)==(1440,256) ? fine_system : CC.prepare_case(n,m;target)
            for r0 in strengths
                trajectory(system,screens,n,m,r0,seed,frames,calibration_dir,divisor,output;fine_system,fine_flat)
            end
            system=nothing;GC.gc(true);CC.CUDA.reclaim()
        end
        fine_system=nothing;screens=nothing;GC.gc(true);CC.CUDA.reclaim()
    end
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    CopperAtmospheric.main(ARGS)
end
