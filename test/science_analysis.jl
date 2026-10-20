using AdaptiveOpticsSim.AlgorithmGraphs: prepared_graph_node
using AdaptiveOpticsSim.Optics: PupilFunction, Source, apply_opd!,
    diffraction_limited_peak, form_direct_image!, intensity_values,
    on_axis_strehl, prepare_direct_imaging, pupil_support

"""Build local PSF analysis products from a clone of the graph's WFS pupil."""
function prepare_science_analysis_fixture(graph, sensor_node::Val;
    wavelength_m::Real)
    wfs_pupil = prepared_graph_node(graph, sensor_node).prepared.input
    open_loop_pupil = PupilFunction(wfs_pupil)
    corrected_pupil = PupilFunction(wfs_pupil)
    source = Source(
        band=:custom,
        wavelength=wavelength_m,
        photon_irradiance=1.0,
        T=Float32,
    )
    open_loop_imaging = prepare_direct_imaging(
        open_loop_pupil, source; zero_padding=2)
    corrected_imaging = prepare_direct_imaging(
        corrected_pupil, source; zero_padding=2)
    open_loop_psf = similar(intensity_values(open_loop_imaging.output))
    corrected_psf = similar(intensity_values(corrected_imaging.output))
    return (; open_loop_pupil, corrected_pupil, open_loop_imaging,
        corrected_imaging, open_loop_psf, corrected_psf,
        support=pupil_support(open_loop_pupil),
        open_loop_peak=diffraction_limited_peak(open_loop_imaging),
        corrected_peak=diffraction_limited_peak(corrected_imaging),
        wavelength_m=Float32(wavelength_m))
end

"""Form normalized PSFs and return native coherent on-axis Strehl values."""
function update_science_analysis!(fixture, open_loop_opd, corrected_opd)
    apply_opd!(fixture.open_loop_pupil, open_loop_opd)
    apply_opd!(fixture.corrected_pupil, corrected_opd)
    open_loop_image = form_direct_image!(fixture.open_loop_imaging)
    corrected_image = form_direct_image!(fixture.corrected_imaging)
    copyto!(fixture.open_loop_psf, intensity_values(open_loop_image))
    copyto!(fixture.corrected_psf, intensity_values(corrected_image))
    fixture.open_loop_psf ./= fixture.open_loop_peak
    fixture.corrected_psf ./= fixture.corrected_peak
    return (
        open_loop=on_axis_strehl(fixture.open_loop_pupil, fixture.wavelength_m),
        corrected=on_axis_strehl(fixture.corrected_pupil, fixture.wavelength_m),
    )
end
