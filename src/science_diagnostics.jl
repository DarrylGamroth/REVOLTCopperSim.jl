using AdaptiveOpticsSim.Optics
using AdaptiveOpticsSim.Optics: TelescopeDefinition, prepare_telescope

mutable struct PreparedScienceDiagnostics{P,I,A,T}
    open_loop_pupil::P
    closed_loop_pupil::P
    open_loop_imaging::I
    closed_loop_imaging::I
    open_loop_psf::A
    closed_loop_psf::A
    open_loop_coherent_real::A
    open_loop_coherent_imag::A
    closed_loop_coherent_real::A
    closed_loop_coherent_imag::A
    diffraction_limited_peak::T
    diffraction_limited_field_power::T
    wavelength_m::T
    open_loop_on_axis_strehl::T
    closed_loop_on_axis_strehl::T
end

"""Return the normalized uncompensated science PSF owned by `diagnostics`."""
@inline open_loop_psf(diagnostics::PreparedScienceDiagnostics) =
    diagnostics.open_loop_psf
"""Return the normalized corrected science PSF owned by `diagnostics`."""
@inline closed_loop_psf(diagnostics::PreparedScienceDiagnostics) =
    diagnostics.closed_loop_psf
"""Return the uncompensated on-axis Strehl ratio."""
@inline open_loop_on_axis_strehl(diagnostics::PreparedScienceDiagnostics) =
    diagnostics.open_loop_on_axis_strehl
"""Return the corrected on-axis Strehl ratio."""
@inline closed_loop_on_axis_strehl(diagnostics::PreparedScienceDiagnostics) =
    diagnostics.closed_loop_on_axis_strehl
"""Return the Boolean pupil support used by the science diagnostics."""
@inline science_pupil_support(diagnostics::PreparedScienceDiagnostics) =
    pupil_support(diagnostics.open_loop_pupil)

"""
    prepare_science_diagnostics()

Prepare host-resident, diffraction-limited-normalized science imaging for the
maintained instrument aperture and science wavelength. This diagnostic owner
accepts uncompensated and corrected pupil OPD maps independently; it is not
part of the detector-frame HIL graph.
"""
function prepare_science_diagnostics()
    definition = TelescopeDefinition(
        resolution=_PUPIL_RESOLUTION,
        diameter=_TELESCOPE_DIAMETER_M,
        central_obstruction=_CENTRAL_OBSTRUCTION_RATIO,
        pupil_reflectivity=1.0,
        revision=1,
        T=Float32,
    )
    telescope = prepare_telescope(definition, HostComputeDevice())
    source = Source(
        band=:custom,
        wavelength=_SCIENCE_WAVELENGTH_M,
        photon_irradiance=1.0,
        T=Float32,
    )
    open_loop_pupil = PupilFunction(telescope)
    closed_loop_pupil = PupilFunction(telescope)
    open_loop_imaging = prepare_direct_imaging(
        open_loop_pupil,
        source;
        zero_padding=2,
    )
    closed_loop_imaging = prepare_direct_imaging(
        closed_loop_pupil,
        source;
        zero_padding=2,
    )
    open_loop_product = form_direct_image!(open_loop_imaging)
    closed_loop_product = form_direct_image!(closed_loop_imaging)
    open_loop_values = intensity_values(open_loop_product)
    amplitude = pupil_amplitude(open_loop_pupil)
    amplitude_power_values = similar(amplitude)
    @. amplitude_power_values = abs2(amplitude)
    amplitude_power = sum(amplitude_power_values)
    diffraction_limited_field_power = abs2(sum(amplitude))
    diffraction_limited_peak = sum(open_loop_values) *
        diffraction_limited_field_power /
        (length(open_loop_values) * amplitude_power)
    isfinite(diffraction_limited_peak) &&
        diffraction_limited_peak > zero(diffraction_limited_peak) || error(
        "the diffraction-limited science peak must be finite and positive",
    )
    open_loop_psf_values = similar(open_loop_values)
    closed_loop_psf_values = similar(intensity_values(closed_loop_product))
    open_loop_coherent_real = similar(amplitude)
    open_loop_coherent_imag = similar(amplitude)
    closed_loop_coherent_real = similar(amplitude)
    closed_loop_coherent_imag = similar(amplitude)
    copyto!(open_loop_psf_values, open_loop_values)
    copyto!(closed_loop_psf_values, intensity_values(closed_loop_product))
    open_loop_psf_values ./= diffraction_limited_peak
    closed_loop_psf_values ./= diffraction_limited_peak
    one_strehl = one(diffraction_limited_peak)
    return PreparedScienceDiagnostics(
        open_loop_pupil,
        closed_loop_pupil,
        open_loop_imaging,
        closed_loop_imaging,
        open_loop_psf_values,
        closed_loop_psf_values,
        open_loop_coherent_real,
        open_loop_coherent_imag,
        closed_loop_coherent_real,
        closed_loop_coherent_imag,
        diffraction_limited_peak,
        diffraction_limited_field_power,
        Float32(_SCIENCE_WAVELENGTH_M),
        one_strehl,
        one_strehl,
    )
end

"""
    update_science_diagnostics!(diagnostics, uncompensated_opd, residual_opd)

Update the open-loop and corrected science products from complete pupil OPD
maps in metres.
"""
function update_science_diagnostics!(
    diagnostics::PreparedScienceDiagnostics,
    uncompensated_opd::AbstractMatrix,
    residual_opd::AbstractMatrix,
)
    apply_opd!(diagnostics.open_loop_pupil, uncompensated_opd)
    apply_opd!(diagnostics.closed_loop_pupil, residual_opd)
    open_loop_product = form_direct_image!(diagnostics.open_loop_imaging)
    closed_loop_product = form_direct_image!(diagnostics.closed_loop_imaging)
    open_loop_values = intensity_values(open_loop_product)
    closed_loop_values = intensity_values(closed_loop_product)
    peak = diagnostics.diffraction_limited_peak
    @. diagnostics.open_loop_psf = open_loop_values / peak
    @. diagnostics.closed_loop_psf = closed_loop_values / peak
    amplitude = pupil_amplitude(diagnostics.open_loop_pupil)
    phase_per_opd = Float32(2 * pi) / diagnostics.wavelength_m
    @. diagnostics.open_loop_coherent_real =
        amplitude * cos(phase_per_opd * uncompensated_opd)
    @. diagnostics.open_loop_coherent_imag =
        amplitude * sin(phase_per_opd * uncompensated_opd)
    @. diagnostics.closed_loop_coherent_real =
        amplitude * cos(phase_per_opd * residual_opd)
    @. diagnostics.closed_loop_coherent_imag =
        amplitude * sin(phase_per_opd * residual_opd)
    open_loop_real = sum(diagnostics.open_loop_coherent_real)
    open_loop_imag = sum(diagnostics.open_loop_coherent_imag)
    closed_loop_real = sum(diagnostics.closed_loop_coherent_real)
    closed_loop_imag = sum(diagnostics.closed_loop_coherent_imag)
    reference_power = diagnostics.diffraction_limited_field_power
    diagnostics.open_loop_on_axis_strehl =
        (open_loop_real^2 + open_loop_imag^2) / reference_power
    diagnostics.closed_loop_on_axis_strehl =
        (closed_loop_real^2 + closed_loop_imag^2) / reference_power
    return diagnostics
end
