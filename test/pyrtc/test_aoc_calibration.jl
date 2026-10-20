using LinearAlgebra: norm, pinv, svdvals
using Test
using LinearAlgebra

if !isdefined(@__MODULE__, :PyRTCProcessHIL)
    include(joinpath(@__DIR__, "pyrtc_process_hil.jl"))
end
using .PyRTCProcessHIL

@testset "REVOLT Copper AOC calibration layout" begin
    responses = Float32[
        3 8
        1 2
        12 9
        4 1
    ]
    poke = 2.0f0
    direct = PyRTCProcessHIL.central_difference_interaction_matrix(responses, poke)
    interaction_matrix = PyRTCProcessHIL.interaction_matrix_from_responses(
        responses,
        poke,
    )

    @test size(direct) == (2, 2)
    @test direct == Float32[0.5 2; 1.5 2]
    @test interaction_matrix == direct
    @test PyRTCProcessHIL.relative_error(direct, interaction_matrix) == 0

    reconstructor = PyRTCProcessHIL.prepare_aoc_reconstructor(
        interaction_matrix,
        2,
        0.05f0,
    )
    @test size(reconstructor.matrix) == (2, 2)
    @test reconstructor.diagnostics.retained_rank == 2
    @test reconstructor.diagnostics.numerical_rank == 2
    @test reconstructor.matrix ≈ pinv(direct; rtol=0.05f0) rtol=3f-6
end

@testset "REVOLT Copper AOC calibration rejects malformed contracts" begin
    @test_throws DimensionMismatch PyRTCProcessHIL.central_difference_interaction_matrix(
        zeros(Float32, 3, 2),
        1.0f0,
    )
    @test_throws DimensionMismatch PyRTCProcessHIL.interaction_matrix_from_responses(
        zeros(Float32, 3, 2),
        1.0f0,
    )
    @test_throws DimensionMismatch PyRTCProcessHIL.prepare_aoc_reconstructor(
        zeros(Float32, 2, 2),
        3,
        0.05f0,
    )
    @test_throws DimensionMismatch PyRTCProcessHIL.validate_control_matrix(
        zeros(Float32, 2, 3),
        (3, 2),
        0.15f0,
    )
    @test_throws ArgumentError PyRTCProcessHIL.validate_control_matrix(
        Float32[NaN 0; 0 0],
        (2, 2),
        0.15f0,
    )
    @test_throws ArgumentError PyRTCProcessHIL.validate_control_matrix(
        zeros(Float32, 2, 2),
        (2, 2),
        Float32(Inf),
    )
end


function synthetic_full_dimension_responses(definition)
    measurement_count = only(definition.signal_shape)
    command_count = definition.command_count
    interaction = zeros(Float32, measurement_count, command_count)
    for command in 1:command_count
        singular_value = command <= 4 ? 1.0f0 :
            command <= 7 ? 0.125f0 : 0.03125f0
        interaction[command, command] = singular_value
    end
    responses = zeros(Float32, 2 * command_count, measurement_count)
    for command in 1:command_count
        @views responses[2 * command - 1, :] .=
            definition.poke .* interaction[:, command]
        @views responses[2 * command, :] .=
            -definition.poke .* interaction[:, command]
    end
    return interaction, responses
end

@testset "REVOLT Copper full-dimension AOC spectrum" begin
    definition = PyRTCProcessHIL.process_definition()
    reference, responses = synthetic_full_dimension_responses(definition)
    interaction = PyRTCProcessHIL.interaction_matrix_from_responses(
        responses,
        definition.poke,
    )
    direct_interaction =
        PyRTCProcessHIL.central_difference_interaction_matrix(
            responses,
            definition.poke,
        )
    interaction_relative_error = PyRTCProcessHIL.relative_error(
        direct_interaction,
        interaction,
    )
    singular_values = svdvals(direct_interaction)
    retained_rank = count(>(maximum(singular_values) * definition.control_rcond),
        singular_values)
    product = PyRTCProcessHIL.prepare_aoc_reconstructor(
        interaction,
        retained_rank,
        definition.control_rcond,
    )
    control = PyRTCProcessHIL.Reconstructors.reconstructor(product)
    reference_control = pinv(
        direct_interaction;
        rtol=definition.control_rcond,
    )
    control_relative_error = norm(control - reference_control) /
        norm(reference_control)

    @test size(interaction) == (1296, 277)
    @test size(control) == (277, 1296)
    @test retained_rank == 7
    @test product.diagnostics.retained_rank == retained_rank
    @test product.diagnostics.numerical_rank == retained_rank
    @test interaction_relative_error < 5f-6
    @test control_relative_error < PyRTCProcessHIL.AOC_RECONSTRUCTOR_RELATIVE_TOLERANCE
    @test interaction ≈ reference rtol=5f-6
    @info "Copper synthetic AOC calibration" dimensions=size(reference) retained_rank interaction_relative_error control_relative_error
end
