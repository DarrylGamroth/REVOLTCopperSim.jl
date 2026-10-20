# Exact nonzero payload from the maintained Copper source artifact:
# - heart/revolt-on-sky origin/copper_integration@8abc4e2,
#   config/dmActuatorMap_277.csv
#   raw SHA-256 76b60effb1786a6cdb37ef3b51c12e34cdc592803f1b1161895b69ee85d51ecc
#
# The Classic source currently has an identical 19-by-19 payload but remains
# independently owned by REVOLTClassicSim. This package retains the Copper
# source-file row order. Command indices
# increase from the last source row to the first source row, with columns as
# the fast actuator axis. Mapping the first row to positive normalized-pupil y
# and the first column to negative x is an explicit provisional simulation convention,
# not a measured pupil registration.
#
# The HEART SRT snapshot contains a separate server-test file named
# `DM277_ActuatorIndex.csv` with a different 277-cell outline. It is retained
# there as a least-squares/test-coordinate fixture and is not the geometry
# authority for this model. The same snapshot's 376-by-277
# `interactionMatrix25102022.fits` agrees with the maintained on-sky signal and
# command ordering, but it does not define command-to-OPD units or a sampled
# influence function.
const _ACTUATOR_AXIS_COUNT = 19
const _ACTUATOR_COUNT = 277
const _PUPIL_ACTUATOR_AXIS_COUNT = 17
const _ROW_WIDTHS_FROM_BOTTOM = (
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    19,
    19,
    19,
    19,
    19,
    19,
    17,
    15,
    13,
    11,
    9,
    7,
)

"""
    actuator_index_map()

Return the exact REVOLT HSDM277 19-by-19 physical-actuator command map. Zero
entries are inactive. Nonzero values are one-based indices into the physical
command vector. Matrix rows and columns retain the exact order of the
maintained Copper source artifact; they do not by themselves define a
normalized-pupil orientation.
"""
function actuator_index_map()
    index_map = zeros(UInt16, _ACTUATOR_AXIS_COUNT, _ACTUATOR_AXIS_COUNT)
    command_index = 1
    for (bottom_row, width) in pairs(_ROW_WIDTHS_FROM_BOTTOM)
        source_row = _ACTUATOR_AXIS_COUNT + 1 - bottom_row
        first_column = (_ACTUATOR_AXIS_COUNT - width) ÷ 2 + 1
        last_column = first_column + width - 1
        for column in first_column:last_column
            index_map[source_row, column] = UInt16(command_index)
            command_index += 1
        end
    end
    return index_map
end

"""
    normalized_pupil_actuator_pitch([T=Float64])

Return the provisional model's adjacent-actuator pitch in normalized pupil
coordinates. REVOLT simulation sources use 17 actuator centres across the
illuminated pupil, so the pitch is `2 / (17 - 1) = 0.125`. This is a model
registration assumption, not a measured HSDM277 pupil registration.
"""
function normalized_pupil_actuator_pitch(
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return T(2) / T(_PUPIL_ACTUATOR_AXIS_COUNT - 1)
end

"""
    provisional_mechanical_coupling([T=Float64])

Return the provisional adjacent-actuator mechanical coupling. The value 0.35
comes from the maintained REVOLT OOPAO ideal-model profile; the older OOMAO
script uses 0.30. This profile deliberately selects 0.35 as a replaceable model
assumption. Neither value is measured HSDM277 influence-function data.
"""
function provisional_mechanical_coupling(
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    return T(0.35)
end

"""
    provisional_gaussian_influence_width([T=Float64])

Convert the provisional pitch and mechanical coupling to the Gaussian width
used by `AdaptiveOpticsSim.Optics.GaussianInfluenceWidth`. Both pitch and width
are in normalized pupil coordinates.
"""
function provisional_gaussian_influence_width(
    ::Type{T}=Float64,
) where {T<:AbstractFloat}
    pitch = normalized_pupil_actuator_pitch(T)
    coupling = provisional_mechanical_coupling(T)
    return pitch / sqrt(-T(2) * log(coupling))
end

"""
    actuator_coordinates([T=Float32])

Return the 2-by-277 HSDM277 command coordinates in normalized pupil
coordinates. Columns follow the exact physical command order in
[`actuator_index_map`](@ref). The 19-actuator physical grid is placed at the
provisional 17-actuator illuminated-pupil pitch, so its outer grid coordinate
is 1.125 rather than the pupil edge at 1.0. Source row one maps to positive y
and source column one maps to negative x as an explicit provisional convention.
"""
function actuator_coordinates(
    ::Type{T}=Float32,
) where {T<:AbstractFloat}
    index_map = actuator_index_map()
    coordinates = Matrix{T}(undef, 2, _ACTUATOR_COUNT)
    pitch = normalized_pupil_actuator_pitch(T)
    centre = (_ACTUATOR_AXIS_COUNT + 1) ÷ 2
    for source_row in axes(index_map, 1), column in axes(index_map, 2)
        command_index = Int(index_map[source_row, column])
        iszero(command_index) && continue
        coordinates[1, command_index] = T(column - centre) * pitch
        coordinates[2, command_index] = T(centre - source_row) * pitch
    end
    return coordinates
end

"""
    actuator_grid_indices([T=Int32])

Return the column-major 19-by-19 grid location of every element in the exact
277-element physical command order. The grid's first axis is increasing
normalized-pupil x and its second axis is increasing normalized-pupil y. These
indices allow a complete physical command to be scattered into the regular
grid used by the separable Gaussian surface evaluation.
"""
function actuator_grid_indices(::Type{T}=Int32) where {T<:Integer}
    index_map = actuator_index_map()
    indices = Vector{T}(undef, _ACTUATOR_COUNT)
    linear_indices = LinearIndices((_ACTUATOR_AXIS_COUNT, _ACTUATOR_AXIS_COUNT))
    for source_row in axes(index_map, 1), column in axes(index_map, 2)
        command_index = Int(index_map[source_row, column])
        iszero(command_index) && continue
        increasing_y_index = _ACTUATOR_AXIS_COUNT + 1 - source_row
        indices[command_index] = T(linear_indices[column, increasing_y_index])
    end
    return indices
end
