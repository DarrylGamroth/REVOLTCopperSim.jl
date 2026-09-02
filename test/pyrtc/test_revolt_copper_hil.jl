using Test

include(joinpath(@__DIR__, "pyrtc_process_hil.jl"))
using .PyRTCProcessHIL

@testset "REVOLT Copper closes through a pyRTC process" begin
    result = PyRTCProcessHIL.run_revolt_copper_validation()
    @test result.system === :revolt_copper
    @test result.frame_shape == (64, 64)
    @test result.signal_length == 1296
    @test result.command_count == 277
    @test result.retained_interaction_rank >= 221
    @test isfinite(result.retained_interaction_condition)
    @test isfinite(result.numerical_interaction_condition)
    @test 0 < result.mean_open_loop_on_axis_strehl < 1
    @test 0.35 < result.mean_closed_loop_on_axis_strehl <= 1
    @test result.improvement >
        PyRTCProcessHIL.REVOLT_COPPER_MINIMUM_STREHL_IMPROVEMENT
    @test result.mean_residual_opd_rms_m <
        0.4 * result.mean_uncompensated_opd_rms_m
    @test isfinite(result.mean_pdm_command_rms_m)
end
