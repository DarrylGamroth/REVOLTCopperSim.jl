const _COMMAND_COUNT = 277
const _PUPIL_RESOLUTION = 480
const _TELESCOPE_DIAMETER_M = 1.22
const _CENTRAL_OBSTRUCTION_RATIO = 0.0
const _SCIENCE_WAVELENGTH_M = 550.0e-9
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

function _graph_definition(profile::Symbol)
    return load_algorithm_graph(
        graph_path(profile);
        bindings=_graph_bindings(profile),
    )
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
    definition = _graph_definition(profile)
    graph = prepare_algorithm_graph(definition; target)
    boundary = prepare_graph_hil_boundary(
        graph;
        command_input=:pdm_command,
        frame_output=:pwfs_frame,
    )
    return (; graph, boundary)
end

function _calibration_detector(production_detector)
    config = production_detector.config
    return emccd_detector_acquisition_node(
        :detector;
        rows=config.rows,
        columns=config.columns,
        binning=config.binning,
        normalized_pupil_sampling=config.normalized_pupil_sampling,
        wavelength_m=config.wavelength_m,
        exposure_duration_s=config.exposure_duration_s,
        quantum_efficiency=config.quantum_efficiency,
        gain=config.gain,
        dark_current_e_per_pixel_s=0,
        bits=config.bits,
        full_well_e=config.full_well_e,
        photon_noise=false,
        readout_noise=false,
        readout_noise_e=0,
        excess_noise_factor=1,
        clock_induced_charge_e_per_pixel_frame=0,
        register_full_well_e=config.register_full_well_e,
        em_gain_min=config.em_gain_range[1],
        em_gain_max=config.em_gain_range[2],
        rng_seed=config.rng_seed,
        photon_rate_schema=config.photon_rate_schema,
        frame_schema=config.frame_schema,
        T=Float32,
    )
end

"""
    prepare_calibration_system(; profile=:grid_gaussian,
        target=HostComputeDevice())

Prepare a flat, noiseless REVOLT Copper graph for a simulation-local Pyramid
interaction matrix. It retains the selected provisional HSDM277 model and the
production four-pupil geometry, but it is not an instrument calibration or a
qualified pixel reconstructor.
"""
function prepare_calibration_system(;
    profile::Symbol=:grid_gaussian,
    target=HostComputeDevice(),
)
    production = _graph_definition(profile)
    length(production.nodes) == 5 || error(
        "the maintained REVOLT Copper HIL graph must contain five nodes",
    )
    pdm = production.nodes[2]
    composition = production.nodes[3]
    pwfs = production.nodes[4]
    detector = _calibration_detector(production.nodes[5])
    uncompensated_opd = zeros(Float32, _PUPIL_RESOLUTION, _PUPIL_RESOLUTION)
    definition = algorithm_graph(
        (pdm, composition, pwfs, detector);
        name=:revolt_copper_hil_calibration,
        inputs=(
            first(production.inputs),
            graph_input(
                :uncompensated_opd,
                :pupil_opd_composition => :uncompensated_opd,
                uncompensated_opd,
            ),
        ),
        outputs=(
            graph_output(:pdm_surface_opd, :pdm => :surface_opd),
            graph_output(:pupil_opd, :pupil_opd_composition => :pupil_opd),
            graph_output(:pwfs_photon_rate, :pwfs => :photon_rate),
            graph_output(:pwfs_frame, :detector => :frame),
        ),
        links=(
            link(
                :pdm => :surface_opd,
                :pupil_opd_composition => :surface_opd,
            ),
            link(:pupil_opd_composition => :pupil_opd, :pwfs => :opd),
            link(:pwfs => :photon_rate, :detector => :photon_rate),
        ),
        parameters=production.parameters,
    )
    graph = prepare_algorithm_graph(definition; target)
    boundary = prepare_graph_hil_boundary(
        graph;
        command_input=:pdm_command,
        frame_output=:pwfs_frame,
    )
    return (; graph, boundary, uncompensated_opd)
end
