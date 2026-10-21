using Test, Statistics
include("../../benchmarks/copper_atmospheric.jl")
const CA=CopperAtmospheric
const CC=CA.CC
CC.BLAS.set_num_threads(1)
CC.AOS.Backends.set_fft_provider_threads!(1)
@testset "Atmospheric convergence host invariants" begin
    f=CA.science_fixture(120)
    @test CA.science(f,zeros(120,120),zeros(277))==(0.,1.)
    rms,strehl=CA.science(f,fill(1e-7,120,120),zeros(277))
    @test rms<1e-20
    @test strehl≈1 atol=1e-12
    s=CA.initialized_screens(1;master=60)
    a=zeros(Float32,120,120)
    CA.render!(a,s,1,.15;master=60)
    b=similar(a);CA.render!(b,s,1,.10;master=60)
    @test CC.relerr(b,a.*Float32((.15/.10)^(5/6)))<5e-7
    @test all(isfinite,a) && std(a)>0
    if get(ENV,"COPPER_TEST_CUDA","0")=="1"
        CC.CUDA.allowscalar(false)
        d=CC.CUDA.zeros(Float32,120,120)
        CA.render!(d,CC.CUDA.CuArray.(s),1,.15;master=60)
        println("render parity ",CC.relerr(Array(d),a))
        wrapped_cpu=similar(a);CA.render!(wrapped_cpu,s,1000,.15;master=60)
        wrapped_gpu=similar(d);CA.render!(wrapped_gpu,CC.CUDA.CuArray.(s),1000,.15;master=60)
        @test CC.relerr(Array(wrapped_gpu),wrapped_cpu)<1e-5
        @test CC.relerr(Array(d),a)<1e-5
        t=CC.AOS.Backends.compute_device(d)
        g=CC.prepare_case(120,64;target=t)
        c=Float32.(sin.(1:277).*1e-8)
        CA.expose(g,d,c)
        dm=Array(CC.graph_output(g.graph,:pdm_surface_opd))
        f.grid[f.indices].=c
        oracle=(f.basis*f.grid*transpose(f.basis)).*f.support
        println("DM parity ",CC.relerr(dm,oracle))
        @test CC.relerr(dm,oracle)<1e-5
        residual=Array(CC.graph_output(g.graph,:pupil_opd))
        @test CC.relerr(residual,Array(d).+dm)<1e-5
    end
end
