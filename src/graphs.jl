const _COMMAND_COUNT = 277
const _GRAPH_DIRECTORY = normpath(joinpath(pkgdir(REVOLTCopperSim), "graphs"))
const _SUPPORTED_PROFILES = (:coordinate_gaussian, :grid_gaussian)

"""Return the number of physical REVOLT Copper HSDM277 command elements."""
command_count() = _COMMAND_COUNT

"""Return the run-immutable deformable-mirror profiles supported by REVOLT Copper."""
supported_profiles() = _SUPPORTED_PROFILES

@inline _graph_filename(::Val{:coordinate_gaussian}) =
    "revolt_copper_hil_coordinate_gaussian.toml"
@inline _graph_filename(::Val{:grid_gaussian}) =
    "revolt_copper_hil_grid_gaussian.toml"

"""
    graph_path([profile=:grid_gaussian])

Return the maintained REVOLT Copper external-RTC graph for `profile`.
`coordinate_gaussian` evaluates the provisional HSDM277 response from actuator
coordinates. `grid_gaussian` uses the equivalent separable regular-grid
evaluation. The profile is fixed before graph preparation.
"""
function graph_path(profile::Symbol=:grid_gaussian)
    profile in _SUPPORTED_PROFILES || throw(ArgumentError(
        "unsupported REVOLT Copper profile '$profile'; expected one of " *
        "$(join(_SUPPORTED_PROFILES, ", "))",
    ))
    return joinpath(_GRAPH_DIRECTORY, _graph_filename(Val(profile)))
end

function _graph_bindings(profile::Symbol)
    pdm_command = zeros(Float32, command_count())
    if profile === :coordinate_gaussian
        return (; pdm_command, pdm_actuator_coordinates=actuator_coordinates())
    elseif profile === :grid_gaussian
        return (; pdm_command, pdm_actuator_grid_indices=actuator_grid_indices())
    end
    graph_path(profile)
    error("unreachable REVOLT Copper profile")
end

"""
    prepare_hil_system(; profile=:grid_gaussian,
        target=HostComputeDevice())

Prepare the atmosphere-backed REVOLT Copper detector graph and its serialized
277-command/64×64-frame HIL boundary. The returned command and frame buffers
remain owned by the prepared boundary.
"""
function prepare_hil_system(;
    profile::Symbol=:grid_gaussian,
    target=HostComputeDevice(),
)
    definition = load_algorithm_graph(
        graph_path(profile);
        bindings=_graph_bindings(profile),
    )
    graph = prepare_algorithm_graph(definition; target)
    boundary = prepare_graph_hil_boundary(
        graph;
        command_input=:pdm_command,
        frame_output=:pwfs_frame,
    )
    return (; graph, boundary)
end
