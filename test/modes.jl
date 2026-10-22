using Test

@testset "Copper numerical modes" begin
    for (mode, resolution, points) in ((:default, 480, 32), (:fast, 240, 96),
        (:accurate, 1440, 256))
        @test REVOLTCopperSim._mode_sampling(mode) ==
            (; mode, pupil_resolution=resolution, modulation_points=points)
    end
    @test_throws ArgumentError REVOLTCopperSim._mode_sampling(:unknown)
    @test_throws ArgumentError REVOLTCopperSim.prepare_hil_system(; mode=:unknown)
    @test_throws ArgumentError REVOLTCopperSim.prepare_calibration_system(; mode=:unknown)
end
