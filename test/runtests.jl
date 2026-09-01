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
