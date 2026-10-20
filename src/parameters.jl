# Run-immutable scientific parameters and product schemas for REVOLTCopper.

const _PUPIL_RESOLUTION = 480
const _TELESCOPE_DIAMETER_M = 1.22
const _CENTRAL_OBSTRUCTION_RATIO = 0.0
const _SCIENCE_WAVELENGTH_M = 550.0e-9
const _ATMOSPHERE_STEP = 0.002
const _PDM_INFLUENCE_WIDTH = 0.08626550214129701
const _EXPOSURE_DURATION_S = 0.002
const _SUPPORTED_PROFILES = (:coordinate_gaussian, :grid_gaussian)
const _ATMOSPHERE_OPD_SCHEMA = "org.revolt.copper.atmosphere-opd-m.f32/1"
const _PDM_COMMAND_SCHEMA = "org.revolt.hsdm277.actuator-surface-opd-m.f32/1"
const _PDM_SURFACE_OPD_SCHEMA = "org.revolt.hsdm277.pdm-surface-opd-m.f32/1"
const _PUPIL_OPD_SCHEMA = "org.revolt.copper.pupil-opd-m.f32/1"
const _PHOTON_RATE_SCHEMA = "org.revolt.copper.pwfs-photon-rate.f32/1"
const _FRAME_SCHEMA = "org.revolt.copper.pwfs-frame.f32/1"
