import AdaptiveOpticsSim
using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.WavefrontSensors
using REVOLTCopperSim
using Test

@testset "REVOLT Copper HSDM277 command geometry" begin
    index_map = actuator_index_map()
    coordinates = actuator_coordinates(Float32)
    grid_indices = actuator_grid_indices(Int32)

    @test command_count() == 277
    @test size(index_map) == (19, 19)
    @test count(!iszero, index_map) == 277
    @test sort(filter(!iszero, vec(index_map))) == UInt16.(1:277)
    @test Tuple(count(!iszero, @view(index_map[row, :])) for row in 1:19) ==
        (7, 9, 11, 13, 15, 17, 19, 19, 19, 19, 19, 19, 19, 17, 15, 13,
            11, 9, 7)
    @test index_map[19, 7:13] == UInt16.(1:7)
    @test index_map[10, :] == UInt16.(130:148)
    @test index_map[1, 7:13] == UInt16.(271:277)
    @test size(coordinates) == (2, 277)
    @test coordinates[:, 1] == Float32[-0.375, -1.125]
    @test coordinates[:, 139] == Float32[0, 0]
    @test coordinates[:, 277] == Float32[0.375, 1.125]
    @test length(unique(grid_indices)) == 277
    @test extrema(grid_indices) == (Int32(7), Int32(355))
    @test normalized_pupil_actuator_pitch(Float32) == 0.125f0
    @test provisional_mechanical_coupling(Float32) == 0.35f0
    @test provisional_gaussian_influence_width(Float64) ≈
        0.08626550214129701
end

@testset "REVOLT Copper calibration preparation" begin
    calibration = prepare_calibration_system()
    @test graph_name(calibration.graph) === :revolt_copper_hil_calibration
    @test size(hil_frame_buffer(calibration.boundary)) == (64, 64)
    @test all(iszero, calibration.uncompensated_opd)
    sequence = step_hil_frame!(calibration.boundary)
    @test sequence == UInt64(1)
    @test all(isfinite, hil_frame_buffer(calibration.boundary))
    @test sum(hil_frame_buffer(calibration.boundary)) > 0

    flat_frame = copy(hil_frame_buffer(calibration.boundary))
    command = hil_command_buffer(calibration.boundary)
    fill!(command, 0.0f0)
    command[139] = 2.0f-8
    adopt_hil_command!(calibration.boundary, sequence)
    sequence = step_hil_frame!(calibration.boundary)
    positive_frame = copy(hil_frame_buffer(calibration.boundary))

    fill!(command, 0.0f0)
    command[139] = -2.0f-8
    adopt_hil_command!(calibration.boundary, sequence)
    @test step_hil_frame!(calibration.boundary) == UInt64(3)
    negative_frame = hil_frame_buffer(calibration.boundary)
    @test positive_frame != flat_frame
    @test negative_frame != flat_frame
    @test positive_frame != negative_frame
    @test all(isfinite, positive_frame)
    @test all(isfinite, negative_frame)
end

@testset "REVOLT Copper deterministic replay" begin
    first_system = prepare_hil_system()
    second_system = prepare_hil_system()
    for expected_sequence in UInt64(1):UInt64(3)
        first_sequence = step_hil_frame!(first_system.boundary)
        second_sequence = step_hil_frame!(second_system.boundary)
        @test first_sequence == expected_sequence
        @test second_sequence == expected_sequence
        @test hil_frame_buffer(first_system.boundary) ==
            hil_frame_buffer(second_system.boundary)
        @test graph_output(first_system.graph, Val(:atmosphere_opd)) ==
            graph_output(second_system.graph, Val(:atmosphere_opd))
        adopt_hil_command!(first_system.boundary, first_sequence)
        adopt_hil_command!(second_system.boundary, second_sequence)
    end
end

@testset "REVOLT Copper science diagnostics" begin
    diagnostics = prepare_science_diagnostics()
    zero_opd = zeros(Float32, 480, 480)
    update_science_diagnostics!(diagnostics, zero_opd, zero_opd)
    @test size(science_pupil_support(diagnostics)) == (480, 480)
    @test size(open_loop_psf(diagnostics)) == (960, 960)
    @test size(closed_loop_psf(diagnostics)) == (960, 960)
    @test open_loop_on_axis_strehl(diagnostics) ≈ 1.0f0
    @test closed_loop_on_axis_strehl(diagnostics) ≈ 1.0f0
    @test maximum(open_loop_psf(diagnostics)) <= 1.001f0
    @test maximum(closed_loop_psf(diagnostics)) <= 1.001f0
end

@testset "REVOLT Copper graph profiles" begin
    @test supported_profiles() == (:coordinate_gaussian, :grid_gaussian)
    @test basename(graph_path()) ==
        "revolt_copper_hil_grid_gaussian.toml"
    @test basename(graph_path(:coordinate_gaussian)) ==
        "revolt_copper_hil_coordinate_gaussian.toml"
    @test_throws ArgumentError graph_path(:unknown)

    for profile in supported_profiles()
        system = prepare_hil_system(; profile)
        graph = system.graph
        boundary = system.boundary
        sequence = step_hil_frame!(boundary)
        @test sequence == UInt64(1)
        @test graph_name(graph) === Symbol("revolt_copper_hil_", profile)
        @test size(hil_frame_buffer(boundary)) == (64, 64)
        @test all(isfinite, hil_frame_buffer(boundary))
        @test sum(hil_frame_buffer(boundary)) > 0
        pwfs_owner = AdaptiveOpticsSim.AlgorithmGraphs.prepared_graph_node(
            graph,
            Val(:pwfs),
        )
        @test pwfs_owner.prepared.plan.propagation.
            modulation_propagation_strategy isa
            WavefrontSensors.PyramidShiftedMaskStrategy
        first_atmosphere_opd = copy(graph_output(graph, Val(:atmosphere_opd)))

        fill!(hil_command_buffer(boundary), 0.0f0)
        hil_command_buffer(boundary)[139] = 5.0f-8
        adopt_hil_command!(boundary, sequence)
        @test step_hil_frame!(boundary) == UInt64(2)
        @test maximum(graph_output(graph, Val(:pdm_surface_opd))) ≈
            5.0f-8 rtol = 5.0f-3
        @test graph_output(graph, Val(:atmosphere_opd)) !=
            first_atmosphere_opd
        @test graph_output(graph, Val(:pupil_opd)) ≈
            graph_output(graph, Val(:atmosphere_opd)) .+
            graph_output(graph, Val(:pdm_surface_opd))
    end
end

if get(ENV, "REVOLT_COPPER_PYRTC_TESTS", "0") == "1"
    include(joinpath(@__DIR__, "pyrtc", "test_revolt_copper_hil.jl"))
end
