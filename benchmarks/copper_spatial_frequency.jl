# Additional accuracy checks outside the calibration-controlled spatial band.
include("copper_convergence.jl")
using .CopperConvergence
const CC = CopperConvergence
function spatial_frequency_rates(system,n)
    axis=(2 .* collect(1:n) .- n .- 1) ./ n
    names=("flat","fourier32_20nm","fourier64_20nm","fourier96_20nm","fourier192_20nm","multiscale100nm")
    rates=Array{Float32}(undef,64,64,length(names))
    for (index,name) in enumerate(names)
        opd=[Float32(if index==1
            0.0
        elseif index<=5
            f=(32,64,96,192)[index-1]
            20e-9*sinpi(f*x)*cospi((f-1)*y)
        else
            # Deterministic logarithmic bands, f^(-5/6) amplitude envelope.
            sum(100e-9*f^(-5/6)*sinpi(f*x+0.37)*cospi((f+1)*y-0.19)
                for f in (1,2,4,8,16,32,64,128,192))
        end) for x in axis,y in axis]
        copyto!(CC.graph_input(system.graph,:uncompensated_opd),opd)
        CC.CUDA.synchronize();CC.step_graph!(system.graph)
        rates[:,:,index] .= Array(CC.graph_output(system.graph,:pwfs_photon_rate))
    end
    return names,rates
end
function main(args)
    directory=abspath(args[1]);mkpath(directory)
    pairs=split(args[2],',')
    CC.CUDA.allowscalar(false);CC.BLAS.set_num_threads(1);CC.AOS.Backends.set_fft_provider_threads!(1)
    target=CC.AOS.Backends.compute_device(CC.CUDA.zeros(Float32,1))
    for pair in pairs
        n,m=parse.(Int,split(pair,':'))
        system=CC.prepare_case(n,m;target)
        names,rates=spatial_frequency_rates(system,n)
        data=Dict("resolution"=>n,"modulation_points"=>m,"probes"=>collect(names),
            "harness_sha256"=>CC.bytes2hex(CC.sha256(read(@__FILE__))),
            "graph_sha256"=>CC.GRAPH_SHA,"base_harness_sha256"=>CC.HARNESS_SHA,
            "rates"=>CC.write_array(directory,"n$(n)-m$(m)-rates",rates))
        CC.persist(joinpath(directory,"n$(n)-m$(m).toml"),data)
        system=nothing;GC.gc(true);CC.CUDA.reclaim()
        println("spatial-frequency checks n=",n," modulation=",m);flush(stdout)
    end
end
main(ARGS)
