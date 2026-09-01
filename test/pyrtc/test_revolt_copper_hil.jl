using Test

include(joinpath(@__DIR__, "pyrtc_process_hil.jl"))
using .PyRTCProcessHIL

@testset "REVOLT Copper exchanges Pyramid frames with a pyRTC process" begin
    result = PyRTCProcessHIL.run_revolt_copper_validation()
    @test result.system === :revolt_copper
    @test result.frame_shape == (64, 64)
    @test result.signal_length == 1296
    @test result.command_count == 277
    @test result.command_indices == (70, 105, 139, 173, 208)
    @test result.probe_rank == length(result.command_indices)
    @test isfinite(result.probe_condition)
    @test all(isfinite, result.response_norms)
    @test all(>(0), result.response_norms)
    @test isfinite(result.atmospheric_signal_norm)
    @test result.atmospheric_signal_norm > 0
end
