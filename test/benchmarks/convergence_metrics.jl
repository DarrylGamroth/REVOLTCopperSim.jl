using Test
include("../../benchmarks/copper_convergence.jl")
using .CopperConvergence
const CC = CopperConvergence
@testset "Copper convergence measurements" begin
    flat = ones(Float32,64,64)
    @test CC.flat_divisor(flat) == 4
    @test all(iszero, CC.raw_slopes(flat,4))
    @test count(CC.SUPPORT) == 716
    for (quadrant, expected) in enumerate(((0.25,0.25),(-0.25,0.25),(-0.25,-0.25),(0.25,-0.25)))
        image = copy(flat)
        r,c=CC.ORIGINS[quadrant]
        image[(c+1):(c+30),(r+1):(r+30)] .+= 1
        slopes=reshape(CC.raw_slopes(image,4),:,2)
        @test all(==(expected[1]), slopes[:,1])
        @test all(==(expected[2]), slopes[:,2])
    end
    @test CC.relerr([1,2],[1,2]) == 0
    @test CC.relerr([0.0],[0.0]) == 0
    @test CC.relerr([1.0],[0.0]) == Inf
    @test CC.relerr([2.0],[1.0]) == 1
    @test CC.opd_probe(120,"piston") == fill(Float32(100e-9),120,120)
    @test maximum(abs,CC.opd_probe(120,"tilt_x_20nm")) < 20e-9
    @test_throws ErrorException CC.opd_probe(120,"unknown")
    mktempdir() do directory
        original=reshape(Float32.(1:24),2,3,4)
        record=CC.write_array(directory,"roundtrip",original)
        @test CC.read_array(directory,record) == original
    end
end
