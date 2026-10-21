using Test
include("../../benchmarks/copper_convergence.jl")
using .CopperConvergence
const CC = CopperConvergence
@testset "Copper CUDA convergence oracle and capture" begin
    CC.CUDA.functional() || error("CUDA hardware required")
    CC.CUDA.allowscalar(false)
    CC.BLAS.set_num_threads(1)
    CC.AOS.Backends.set_fft_provider_threads!(1)
    target=CC.AOS.Backends.compute_device(CC.CUDA.zeros(Float32,1))
    gpu=CC.prepare_case(120,64;target)
    host=CC.AOS.Backends.HostComputeDevice()
    definition=CC.REVOLTCopperSim._graph_definition(:grid_gaussian,host;
        calibration=true,pupil_resolution=120,modulation_points=64)
    graph=CC.prepare_algorithm_graph(definition;target=host,execution=CC.StreamGraphExecution())
    boundary=CC.prepare_graph_hil_boundary(graph;command_input=:pdm_command,frame_output=:pwfs_frame)
    cpu=(;graph,boundary)
    gpu_rates=CC.probe_rates(gpu,120)
    cpu_rates=CC.probe_rates(cpu,120)
    errors=[CC.relerr(gpu_rates[:,:,k],cpu_rates[:,:,k]) for k in eachindex(CC.PROBES)]
    @test all(<(2e-4),errors)
    divisor=CC.flat_divisor(gpu_rates[:,:,1])
    columns=[1,7,50,100,130,139,148,177,228,271,277]
    gpu_matrix=CC.calibration_response(gpu,divisor;columns)
    cpu_matrix=CC.calibration_response(cpu,divisor;columns)
    matrix_error=CC.relerr(gpu_matrix,cpu_matrix)
    @test matrix_error < 5e-4
    @test CC.captured_graph_node_count(gpu.graph)==4
    @test CC.timings(CC.step_graph!,gpu.graph;samples=16)["host_bytes"]==zeros(Int,16)
    CC.reset_hil_boundary!(gpu.boundary)
    @test CC.timings(CC.cycle!,gpu.boundary;samples=16)["host_bytes"]==zeros(Int,16)
    println("Maximum CPU/CUDA photon-rate relative L2: ",maximum(errors))
    println("CPU/CUDA selected calibration relative L2: ",matrix_error)
end
