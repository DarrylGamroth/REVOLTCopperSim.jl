# REVOLT Copper scientific plant graph prescription.
#
# The graph owns one explicitly timed five-layer atmosphere and advances it by
# 2 ms for every complete frame. A serialized RTC adapter places one complete
# PDM command in the caller-owned command buffer before each graph step. This
# graph applies that command through the same explicitly provisional HSDM277
# continuous-facesheet model as REVOLT Classic, composes the surface OPD with
# the atmospheric OPD, forms one modulated Pyramid photon-rate frame, and
# completes one noisy EMCCD frame for publication through a transport adapter
# such as PipeWireAO. The external RTC owns pixel processing, reconstruction,
# control, and command production.
#
# Model authority is deliberately layered:
# - REVOLT/Julia/contracts/revolt/common.toml supplies the 1.22 m, 480-sample
#   telescope and 2 ms simulation sample period.
# - REVOLT/Julia/contracts/revolt/pwfs.toml supplies the modulated Pyramid
#   modulation radius/point count and visible-source flux scaling.
#   Estimator thresholds and pupil selection belong to the external RTC.
#   The effective V magnitude below includes its 0.5 flux scale.
# - REVOLT/Julia/contracts/revolt/cameras.toml supplies the iXon EMCCD values.
# - heart/revolt-rtc/config/copper_config_sim.yaml supplies the operational
#   64-by-64 WFS frame and 30-by-30 Pyramid-pupil dimensions.
# - `REVOLTCopperSim.actuator_coordinates` reproduces the exact
#   physical-actuator command map and the explicitly provisional
#   17-across-pupil, 0.35-coupling Gaussian response used until measured
#   HSDM277 influence functions exist.
#
# The symmetric AOS pupil-image placement is transport-shape compatible but is
# not a claim that the current measured pwfsRoiOffsets_64.csv registration or
# an operational pixel reconstructor is calibrated to this simulated image.
# The fixed modulation cycle explicitly selects shifted-mask propagation for
# throughput. It preserves the quadrature and output normalization but is a
# sampled detector-product approximation to the pupil-tilt reference strategy.
# The PDM input contains unit-peak actuator surface-OPD coefficients in metres,
# not ALPAO normalized hardware commands.

"""Return the number of physical REVOLT Copper HSDM277 command elements."""
command_count() = _ACTUATOR_COUNT

"""Return the run-immutable deformable-mirror profiles supported by REVOLT Copper."""
supported_profiles() = _SUPPORTED_PROFILES

function _require_profile(profile::Symbol)
    profile in _SUPPORTED_PROFILES || throw(ArgumentError(
        "unsupported REVOLT Copper profile '$profile'; expected one of " *
        "$(join(_SUPPORTED_PROFILES, ", "))",
    ))
    return nothing
end

function _copy_to_target(target, values::AbstractArray)
    target_values = allocate_device_array(target, eltype(values), size(values)...)
    copyto!(target_values, values)
    return target_values
end

function _pdm_node(::Val{:coordinate_gaussian}, telescope)
    return gaussian_deformable_mirror_surface_node(
        :pdm; telescope..., actuator_count=command_count(),
        influence_width=_PDM_INFLUENCE_WIDTH,
        pdm_command_schema=_PDM_COMMAND_SCHEMA,
        surface_opd_schema=_PDM_SURFACE_OPD_SCHEMA,
        actuator_coordinates_schema="org.revolt.hsdm277.normalized-pupil-actuator-coordinates.f32/1",
    )
end

function _pdm_node(::Val{:grid_gaussian}, telescope)
    return grid_gaussian_deformable_mirror_surface_node(
        :pdm; telescope..., actuator_count=command_count(),
        actuator_axis_count=19, actuator_pitch=0.125,
        influence_width=_PDM_INFLUENCE_WIDTH,
        pdm_command_schema=_PDM_COMMAND_SCHEMA,
        surface_opd_schema=_PDM_SURFACE_OPD_SCHEMA,
        actuator_grid_indices_schema="org.revolt.hsdm277.actuator-grid-indices.i32/1",
    )
end

_pdm_parameter(::Val{:coordinate_gaussian}, target) = sparse_parameter(
    :pdm => :actuator_coordinates, _copy_to_target(target, actuator_coordinates()))
_pdm_parameter(::Val{:grid_gaussian}, target) = sparse_parameter(
    :pdm => :actuator_grid_indices, _copy_to_target(target, actuator_grid_indices()))

function _graph_definition(
    profile::Symbol,
    target;
    calibration::Bool=false,
    atmosphere_step::Real=_ATMOSPHERE_STEP,
    exposure_duration_s::Real=_EXPOSURE_DURATION_S,
)
    _require_profile(profile)
    telescope = (;
        resolution=_PUPIL_RESOLUTION,
        telescope_diameter_m=_TELESCOPE_DIAMETER_M,
        central_obstruction_ratio=_CENTRAL_OBSTRUCTION_RATIO,
        pupil_reflectivity=1.0, aperture_revision=1, T=Float32,
    )
    pdm = _pdm_node(Val(profile), telescope)
    pupil_opd_composition = pupil_opd_composition_node(
        :pupil_opd_composition;
        resolution=_PUPIL_RESOLUTION,
        uncompensated_opd_schema=_ATMOSPHERE_OPD_SCHEMA,
        surface_opd_schema=_PDM_SURFACE_OPD_SCHEMA,
        pupil_opd_schema=_PUPIL_OPD_SCHEMA,
        T=Float32,
    )
    pwfs = pyramid_rate_node(
        :pwfs; telescope...,
        pupil_samples=30,
        modulation=5.0,
        modulation_points=32,
        modulation_propagation_strategy=PyramidShiftedMaskStrategy(),
        diffraction_padding=2,
        psf_centering=true,
        n_pix_separation=2,
        n_pix_edge=1,
        binning=1,
        source_band=:V,
        source_magnitude=5.752574989159953,
        source_wavelength_m=5.5e-7,
        source_separation_arcsec=0.0,
        source_position_angle_deg=0.0,
        opd_schema=_PUPIL_OPD_SCHEMA,
        photon_rate_schema=_PHOTON_RATE_SCHEMA,
    )
    detector = emccd_detector_acquisition_node(
        :detector;
        rows=64,
        columns=64,
        binning=1,
        normalized_pupil_sampling=0.03333333333333333,
        wavelength_m=5.5e-7,
        exposure_duration_s=exposure_duration_s,
        quantum_efficiency=0.95,
        gain=1.0,
        dark_current_e_per_pixel_s=calibration ? 0 : 20.0,
        bits=14,
        full_well_e=16000.0,
        photon_noise=!calibration,
        readout_noise=!calibration,
        readout_noise_e=calibration ? 0 : 1.0,
        excess_noise_factor=1.0,
        clock_induced_charge_e_per_pixel_frame=0.0,
        rng_seed=0,
        photon_rate_schema=_PHOTON_RATE_SCHEMA,
        frame_schema=_FRAME_SCHEMA,
        T=Float32,
    )

    command = graph_input(:pdm_command, :pdm => :pdm_command,
        _copy_to_target(target, zeros(Float32, command_count())))
    pdm_output = graph_output(:pdm_surface_opd, :pdm => :surface_opd)
    sensor_outputs = (
        graph_output(:pupil_opd, :pupil_opd_composition => :pupil_opd),
        graph_output(:pwfs_photon_rate, :pwfs => :photon_rate),
        graph_output(:pwfs_frame, :detector => :frame),
    )
    surface_link = link(:pdm => :surface_opd, :pupil_opd_composition => :surface_opd)
    sensor_links = (
        link(:pupil_opd_composition => :pupil_opd, :pwfs => :opd),
        link(:pwfs => :photon_rate, :detector => :photon_rate),
    )
    sensor_nodes = (pdm, pupil_opd_composition, pwfs, detector)
    if calibration
        uncompensated_opd = _copy_to_target(target,
            zeros(Float32, _PUPIL_RESOLUTION, _PUPIL_RESOLUTION))
        nodes = sensor_nodes
        inputs = (command, graph_input(:uncompensated_opd,
            :pupil_opd_composition => :uncompensated_opd, uncompensated_opd))
        outputs = (pdm_output, sensor_outputs...)
        links = (surface_link, sensor_links...)
        name = :revolt_copper_hil_calibration
    else
        atmosphere = multilayer_atmosphere_opd_node(
            :atmosphere; telescope...,
            r0=0.15,
            reference_wavelength_m=5.0e-7,
            L0=30.0,
            fractional_cn2=[0.45, 0.1, 0.1, 0.25, 0.1],
            wind_speed=[5.0, 5.0, 7.0, 10.0, 10.0],
            wind_direction_deg=[0.0, 72.0, 144.0, 216.0, 288.0],
            altitude=[0.0, 1000.0, 5000.0, 10000.0, 12000.0],
            layer_ids=[:revolt_ground, :revolt_1km, :revolt_5km, :revolt_10km, :revolt_12km],
            atmosphere_step=atmosphere_step,
            rng_seed=1,
            atmosphere_opd_schema=_ATMOSPHERE_OPD_SCHEMA,
        )
        nodes = (atmosphere, sensor_nodes...)
        inputs = (command,)
        outputs = (pdm_output,
            graph_output(:atmosphere_opd, :atmosphere => :atmosphere_opd),
            sensor_outputs...)
        links = (surface_link,
            link(:atmosphere => :atmosphere_opd,
                :pupil_opd_composition => :uncompensated_opd),
            sensor_links...)
        name = Symbol(:revolt_copper_hil_, profile)
    end
    return algorithm_graph(nodes; name, inputs, outputs, links,
        parameters=(_pdm_parameter(Val(profile), target),))
end

function _prepare_system(profile, target, execution; calibration::Bool=false)
    definition = _graph_definition(profile, target; calibration)
    graph = prepare_algorithm_graph(definition; target, execution)
    boundary = prepare_graph_hil_boundary(graph;
        command_input=:pdm_command, frame_output=:pwfs_frame)
    frame_period_s = Float32(_ATMOSPHERE_STEP)
    if calibration
        uncompensated_opd = graph_input(graph, :uncompensated_opd)
        return (; graph, boundary, frame_period_s, uncompensated_opd)
    end
    return (; graph, boundary, frame_period_s)
end

"""
    prepare_hil_system(; profile=:grid_gaussian,
        target=HostComputeDevice(), execution=StreamGraphExecution())

Prepare the atmosphere-backed REVOLT Copper Julia graph and its serialized
277-command/64×64-frame HIL boundary. `profile` selects the coordinate or
separable-grid implementation of the same provisional HSDM277 response.
`frame_period_s` records the run-immutable model cadence independently of the
detector exposure. Command and frame buffers remain owned by the boundary.
"""
function prepare_hil_system(;
    profile::Symbol=:grid_gaussian,
    target=HostComputeDevice(),
    execution=StreamGraphExecution(),
)
    return _prepare_system(profile, target, execution)
end

"""
    prepare_calibration_system(; profile=:grid_gaussian,
        target=HostComputeDevice(), execution=StreamGraphExecution())

Prepare the same REVOLT Copper prescription with a flat `uncompensated_opd`
input and noiseless detector for a simulation-local interaction matrix. The
provisional HSDM277 model and sensor geometry are shared with production; this
is not a measured instrument calibration or a qualified pixel reconstructor.
"""
function prepare_calibration_system(;
    profile::Symbol=:grid_gaussian,
    target=HostComputeDevice(),
    execution=StreamGraphExecution(),
)
    return _prepare_system(profile, target, execution; calibration=true)
end
