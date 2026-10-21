using AdaptiveOpticsSim
using AdaptiveOpticsSim.Backends: HostComputeDevice
using AdaptiveOpticsSim.AlgorithmGraphs
using REVOLTCopperSim
using Test

@testset "private convergence graph preparation" begin
    target = HostComputeDevice()
    execution = StreamGraphExecution()

    default_definition = REVOLTCopperSim._graph_definition(
        :grid_gaussian, target; calibration=true)
    pwfs_definition = only(filter(node->AdaptiveOpticsSim.AlgorithmGraphs._node_name(node) === :pwfs, default_definition.nodes))
    @test pwfs_definition.config.modulation_points == 32
    @test pwfs_definition.config.telescope.resolution == 480
    default_graph = prepare_algorithm_graph(default_definition; target, execution)
    default_boundary = prepare_graph_hil_boundary(default_graph;
        command_input=:pdm_command, frame_output=:pwfs_frame)

    public_system = prepare_calibration_system(; target, execution)
    @test size(hil_frame_buffer(default_boundary)) ==
        size(hil_frame_buffer(public_system.boundary)) == (64, 64)
    @test size(graph_output(default_graph, Val(:pupil_opd))) == (480, 480)
    @test step_hil_frame!(default_boundary) == UInt64(1)
    @test step_hil_frame!(public_system.boundary) == UInt64(1)
    @test hil_frame_buffer(default_boundary) == hil_frame_buffer(public_system.boundary)

    alternate_definition = REVOLTCopperSim._graph_definition(
        :grid_gaussian, target; calibration=true,
        pupil_resolution=60, modulation_points=4)
    alternate_graph = prepare_algorithm_graph(alternate_definition; target, execution)
    alternate_boundary = prepare_graph_hil_boundary(alternate_graph;
        command_input=:pdm_command, frame_output=:pwfs_frame)
    @test size(hil_frame_buffer(alternate_boundary)) == (64, 64)
    @test length(hil_command_buffer(alternate_boundary)) == 277
    @test size(graph_output(alternate_graph, Val(:pupil_opd))) == (60, 60)
end
