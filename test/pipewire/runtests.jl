import REVOLTCopperSim
using AdaptiveOpticsSim.AlgorithmGraphs
using AdaptiveOpticsSim.Backends: HostComputeDevice
using AdaptiveOpticsSimPipeWireHIL
using PipeWireAO
using Test

@testset "REVOLT Copper optional PipeWire configuration" begin
    for profile in REVOLTCopperSim.supported_profiles()
        system = REVOLTCopperSim.prepare_hil_system(; profile)
        configuration = REVOLTCopperSim.pipewire_configuration(system, "unconnected-copper-test")
        @test configuration.remote == "unconnected-copper-test"
        @test configuration.frame_node_name == "revolt-copper-sim-wfs"
        @test configuration.command_node_name == "revolt-copper-sim-command"
        @test configuration.frame_schema == "org.revolt.copper.pwfs-frame.f32/1"
        @test configuration.command_schema == "org.revolt.hsdm277.actuator-surface-opd-m.f32/1"
        @test configuration.frame_encoding === :float32
        @test configuration.command_scale === 1f0
        @test (configuration.rate.num, configuration.rate.denom) == (500, 1)
        @test configuration.exposure_duration_ns == 2_000_000
        @test configuration.timeout_ns == 5_000_000_000
        @test size(hil_frame_buffer(system.boundary)) == (64, 64)
        @test graph_step_sequence(system.graph) == 0
        @test hil_boundary_status(system.boundary).active_command_sequence == 0
        explicit = REVOLTCopperSim.pipewire_configuration(system, "unconnected-copper-test";
            frame_node_name="explicit-frame", command_node_name="explicit-command",
            timeout_ns=123_456_789,
            frame_encoding=:uint16, frame_schema="org.calculon.ao.raw-detector-pixels/1",
            command_schema="org.calculon.ao.demanded-pdm-command/1",
            command_scale=1f-6)
        @test explicit.frame_node_name == "explicit-frame"
        @test explicit.command_node_name == "explicit-command"
        @test explicit.timeout_ns == 123_456_789
        @test explicit.frame_encoding === :uint16
        @test explicit.frame_schema == "org.calculon.ao.raw-detector-pixels/1"
        @test explicit.command_schema == "org.calculon.ao.demanded-pdm-command/1"
        @test explicit.command_scale === 1f-6
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(system, "test";
            frame_encoding=:uint16)
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(system, "test";
            frame_encoding=:uint16, frame_schema=configuration.frame_schema)
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(system, "test";
            frame_schema="test.wrong.f32/1")
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(system, "test";
            command_scale=1f-6)
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(system, "test";
            command_schema="org.calculon.ao.demanded-pdm-command/1")
        wrong_boundary = prepare_graph_hil_boundary(system.graph;
            command_input=:pdm_command, frame_output=:pupil_opd)
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(
            (; system.graph, boundary=wrong_boundary), "test")
        @test graph_step_sequence(system.graph) == 0
    end

    # A changed model-time step and detector exposure must be read from the
    # already resolved plant. No filename defaults or positional node lookup.
    mktempdir() do directory
        graph_text = replace(read(REVOLTCopperSim.graph_path(), String),
            "atmosphere_step = 0.002" => "atmosphere_step = 0.003",
            "exposure_duration_s = 0.002" => "exposure_duration_s = 0.001")
        path = joinpath(directory, "modified.toml")
        write(path, graph_text)
        definition = load_algorithm_graph(path;
            bindings=REVOLTCopperSim._graph_bindings(:grid_gaussian, HostComputeDevice()))
        graph = prepare_algorithm_graph(definition)
        boundary = prepare_graph_hil_boundary(graph;
            command_input=:pdm_command, frame_output=:pwfs_frame)
        configuration = REVOLTCopperSim.pipewire_configuration((; graph, boundary), "test")
        @test (configuration.rate.num, configuration.rate.denom) == (1000, 3)
        @test configuration.exposure_duration_ns == 1_000_000
        @test graph_step_sequence(graph) == 0
        other = REVOLTCopperSim.prepare_hil_system()
        @test_throws ArgumentError REVOLTCopperSim.pipewire_configuration(
            (; graph=other.graph, boundary), "test")
    end
end
