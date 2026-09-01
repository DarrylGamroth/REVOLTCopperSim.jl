"""
Native Julia endpoint for the maintained v1 compatibility contract around
pyRTC's current Linux `ImageSHM` convention.

Each stream consists of one POSIX shared-memory payload and one ten-`Float64`
`<name>_meta` segment. Metadata contains write count, wall-clock publication
time, payload bytes, NumPy dtype code, and six dimension slots. This adapter
admits vectors and matrices; matrices are copied in NumPy C order at the
boundary.

The upstream convention has one overwriteable slot and no in-progress marker.
Consequently, coherent reads require application-level single-writer lockstep:
the producer must not overwrite a payload until the consumer has completed the
corresponding exchange. `read_next!` rejects metadata that changes during its
copy, but cannot make an unconstrained upstream writer race-free.
"""
module PyRTCSharedMemory

using UnsafeArrays: UnsafeArray

export PyRTCSharedMemoryError
export PyRTCStream
export close
export create_stream
export open_stream
export publish!
export read_next!
export stream_count
export stream_shape
export unlink!

const METADATA_LENGTH = 10
const METADATA_BYTES = METADATA_LENGTH * sizeof(Float64)
const MAX_DIMENSIONS = METADATA_LENGTH - 4
const SUPPORTED_DIMENSIONS = 2
const MAX_EXACT_FLOAT64_INTEGER = UInt64(1) << 53

const O_RDONLY = Cint(0)
const O_RDWR = Cint(2)
const O_CREAT = Cint(64)
const O_EXCL = Cint(128)
const OWNER_PERMISSIONS = Cuint(0o600)
const PROT_READ = Cint(1)
const PROT_WRITE = Cint(2)
const MAP_SHARED = Cint(1)
const SEEK_END = Cint(2)

struct PyRTCSharedMemoryError <: Exception
    operation::Symbol
    stream::String
    detail::String
end

function Base.showerror(io::IO, error::PyRTCSharedMemoryError)
    print(
        io,
        "pyRTC shared-memory ",
        error.operation,
        " failed for '",
        error.stream,
        "': ",
        error.detail,
    )
end

mutable struct SharedMapping
    name::String
    address::Ptr{UInt8}
    nbytes::Int
    closed::Bool
end

mutable struct PyRTCStream{T,N}
    name::String
    shape::NTuple{N,Int}
    payload_mapping::SharedMapping
    metadata_mapping::SharedMapping
    payload::UnsafeArray{T,1}
    metadata::UnsafeArray{Float64,1}
    owner::Bool
    unlinked::Bool
    closed::Bool
    write_count::UInt64
    last_read_count::UInt64
    last_read_timestamp::Float64
end

stream_shape(stream::PyRTCStream) = stream.shape

@inline function _system_detail()
    error_number = Base.Libc.errno()
    return "$(Base.Libc.strerror(error_number)) (errno=$error_number)"
end

@inline function _throw_system_error(operation::Symbol, name::AbstractString)
    throw(PyRTCSharedMemoryError(operation, String(name), _system_detail()))
end

function _validate_stream_name(name::AbstractString)
    value = String(name)
    isempty(value) && throw(PyRTCSharedMemoryError(
        :validate,
        value,
        "stream name must not be empty",
    ))
    ncodeunits(value) <= 240 || throw(PyRTCSharedMemoryError(
        :validate,
        value,
        "stream name must contain at most 240 bytes",
    ))
    all(character -> isascii(character) &&
        (isletter(character) || isdigit(character) ||
         character in ('_', '-', '.')), value) ||
        throw(PyRTCSharedMemoryError(
            :validate,
            value,
            "stream name may contain only ASCII letters, digits, '_', '-', and '.'",
        ))
    return value
end

@inline _metadata_name(name::AbstractString) = String(name) * "_meta"
@inline _posix_name(name::AbstractString) = "/" * String(name)

function _checked_element_count(name::AbstractString, shape::NTuple{N,Int}) where {N}
    1 <= N <= SUPPORTED_DIMENSIONS || throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "the native adapter supports vectors and matrices",
    ))
    count = 1
    for dimension in shape
        dimension > 0 || throw(PyRTCSharedMemoryError(
            :validate,
            String(name),
            "all stream dimensions must be positive",
        ))
        count, overflow = Base.Checked.mul_with_overflow(count, dimension)
        overflow && throw(PyRTCSharedMemoryError(
            :validate,
            String(name),
            "stream element count overflows Int",
        ))
    end
    return count
end

function _checked_payload_bytes(
    name::AbstractString,
    ::Type{T},
    shape::NTuple{N,Int},
) where {T,N}
    isbitstype(T) || throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "stream element type must be isbits",
    ))
    count = _checked_element_count(name, shape)
    nbytes, overflow = Base.Checked.mul_with_overflow(count, sizeof(T))
    overflow && throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "stream byte count overflows Int",
    ))
    UInt64(nbytes) <= MAX_EXACT_FLOAT64_INTEGER || throw(
        PyRTCSharedMemoryError(
            :validate,
            String(name),
            "stream byte count cannot be represented exactly in pyRTC v1 metadata",
        ),
    )
    return nbytes
end

@inline pyrtc_dtype_code(::Type{Int8}) = 0
@inline pyrtc_dtype_code(::Type{Int16}) = 1
@inline pyrtc_dtype_code(::Type{Int32}) = 2
@inline pyrtc_dtype_code(::Type{Int64}) = 3
@inline pyrtc_dtype_code(::Type{UInt8}) = 4
@inline pyrtc_dtype_code(::Type{UInt16}) = 5
@inline pyrtc_dtype_code(::Type{UInt32}) = 6
@inline pyrtc_dtype_code(::Type{UInt64}) = 7
@inline pyrtc_dtype_code(::Type{Float16}) = 8
@inline pyrtc_dtype_code(::Type{Float32}) = 9
@inline pyrtc_dtype_code(::Type{Float64}) = 10
@inline pyrtc_dtype_code(::Type{ComplexF32}) = 11
@inline pyrtc_dtype_code(::Type{ComplexF64}) = 12
@inline pyrtc_dtype_code(::Type{Bool}) = 13

function pyrtc_dtype_code(::Type{T}) where {T}
    throw(PyRTCSharedMemoryError(
        :validate,
        string(T),
        "element type is not supported by the numeric pyRTC v1 adapter",
    ))
end

function _close_file_descriptor(file_descriptor::Cint)
    file_descriptor < 0 && return nothing
    ccall(:close, Cint, (Cint,), file_descriptor)
    return nothing
end

function _unlink_name(name::AbstractString; missing_ok::Bool)
    status = ccall(:shm_unlink, Cint, (Cstring,), _posix_name(name))
    status == 0 && return nothing
    error_number = Base.Libc.errno()
    missing_ok && error_number == Base.Libc.ENOENT && return nothing
    throw(PyRTCSharedMemoryError(
        :unlink,
        String(name),
        "$(Base.Libc.strerror(error_number)) (errno=$error_number)",
    ))
end

function _map_file_descriptor(
    file_descriptor::Cint,
    name::AbstractString,
    nbytes::Int,
)
    address = ccall(
        :mmap,
        Ptr{Cvoid},
        (Ptr{Cvoid}, Csize_t, Cint, Cint, Cint, Int64),
        C_NULL,
        nbytes,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        file_descriptor,
        0,
    )
    reinterpret(Int, address) == -1 && _throw_system_error(:map, name)
    return SharedMapping(
        String(name),
        convert(Ptr{UInt8}, address),
        nbytes,
        false,
    )
end

function _create_mapping(name::AbstractString, nbytes::Int)
    file_descriptor = ccall(
        :shm_open,
        Cint,
        (Cstring, Cint, Cuint),
        _posix_name(name),
        O_RDWR | O_CREAT | O_EXCL,
        OWNER_PERMISSIONS,
    )
    file_descriptor < 0 && _throw_system_error(:create, name)
    mapping = nothing
    try
        status = ccall(
            :ftruncate,
            Cint,
            (Cint, Int64),
            file_descriptor,
            nbytes,
        )
        status == 0 || _throw_system_error(:resize, name)
        mapping = _map_file_descriptor(file_descriptor, name, nbytes)
        bytes = UnsafeArray(
            mapping.address,
            (mapping.nbytes,),
        )
        fill!(bytes, 0x00)
        return mapping
    catch
        if !isnothing(mapping)
            _close_mapping_noexcept!(mapping)
        end
        _unlink_name(name; missing_ok=true)
        rethrow()
    finally
        _close_file_descriptor(file_descriptor)
    end
end

function _open_mapping(name::AbstractString, expected_bytes::Int)
    file_descriptor = ccall(
        :shm_open,
        Cint,
        (Cstring, Cint, Cuint),
        _posix_name(name),
        O_RDWR,
        OWNER_PERMISSIONS,
    )
    file_descriptor < 0 && _throw_system_error(:open, name)
    try
        actual_bytes = ccall(
            :lseek,
            Int64,
            (Cint, Int64, Cint),
            file_descriptor,
            0,
            SEEK_END,
        )
        actual_bytes >= 0 || _throw_system_error(:inspect, name)
        actual_bytes == expected_bytes || throw(PyRTCSharedMemoryError(
            :validate,
            String(name),
            "segment has $actual_bytes bytes; expected $expected_bytes",
        ))
        return _map_file_descriptor(
            file_descriptor,
            name,
            expected_bytes,
        )
    finally
        _close_file_descriptor(file_descriptor)
    end
end

function _close_mapping!(mapping::SharedMapping)
    mapping.closed && return nothing
    status = ccall(
        :munmap,
        Cint,
        (Ptr{Cvoid}, Csize_t),
        mapping.address,
        mapping.nbytes,
    )
    status == 0 || _throw_system_error(:unmap, mapping.name)
    mapping.address = C_NULL
    mapping.closed = true
    return nothing
end

function _close_mapping_noexcept!(mapping::SharedMapping)
    mapping.closed && return nothing
    ccall(
        :munmap,
        Cint,
        (Ptr{Cvoid}, Csize_t),
        mapping.address,
        mapping.nbytes,
    )
    mapping.address = C_NULL
    mapping.closed = true
    return nothing
end

function _require_open(stream::PyRTCStream, operation::Symbol)
    stream.closed && throw(PyRTCSharedMemoryError(
        operation,
        stream.name,
        "stream is closed",
    ))
    return nothing
end

function _metadata_integer(
    value::Float64,
    name::AbstractString,
    field::AbstractString;
    allow_zero::Bool,
)
    isfinite(value) || throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "$field must be finite",
    ))
    minimum = allow_zero ? 0.0 : 1.0
    minimum <= value <= Float64(MAX_EXACT_FLOAT64_INTEGER) || throw(
        PyRTCSharedMemoryError(
            :validate,
            String(name),
            "$field is outside the pyRTC v1 exact-integer range",
        ),
    )
    value == trunc(value) || throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "$field must be an integer-valued Float64",
    ))
    return UInt64(value)
end

function _validate_metadata!(
    metadata::UnsafeArray{Float64,1},
    name::AbstractString,
    ::Type{T},
    shape::NTuple{N,Int},
    expected_bytes::Int,
) where {T,N}
    _metadata_integer(
        metadata[1],
        name,
        "write count";
        allow_zero=true,
    )
    isfinite(metadata[2]) && metadata[2] >= 0 || throw(
        PyRTCSharedMemoryError(
            :validate,
            String(name),
            "publication time must be finite and nonnegative",
        ),
    )
    recorded_bytes = _metadata_integer(
        metadata[3],
        name,
        "payload byte count";
        allow_zero=false,
    )
    recorded_bytes == UInt64(expected_bytes) || throw(
        PyRTCSharedMemoryError(
            :validate,
            String(name),
            "metadata declares $recorded_bytes payload bytes; expected $expected_bytes",
        ),
    )
    recorded_dtype = _metadata_integer(
        metadata[4],
        name,
        "dtype code";
        allow_zero=true,
    )
    expected_dtype = UInt64(pyrtc_dtype_code(T))
    recorded_dtype == expected_dtype || throw(PyRTCSharedMemoryError(
        :validate,
        String(name),
        "metadata dtype code is $recorded_dtype; expected $expected_dtype for $T",
    ))
    for dimension_index in 1:MAX_DIMENSIONS
        raw_dimension = metadata[4 + dimension_index]
        expected_dimension = dimension_index <= N ? shape[dimension_index] : 0
        dimension = _metadata_integer(
            raw_dimension,
            name,
            "shape dimension $dimension_index";
            allow_zero=true,
        )
        dimension == UInt64(expected_dimension) || throw(
            PyRTCSharedMemoryError(
                :validate,
                String(name),
                "metadata dimension $dimension_index is $dimension; expected $expected_dimension",
            ),
        )
    end
    return nothing
end

function _stream_from_mappings(
    name::String,
    ::Type{T},
    shape::NTuple{N,Int},
    payload_mapping::SharedMapping,
    metadata_mapping::SharedMapping;
    owner::Bool,
) where {T,N}
    element_count = _checked_element_count(name, shape)
    payload = UnsafeArray(
        convert(Ptr{T}, payload_mapping.address),
        (element_count,),
    )
    metadata = UnsafeArray(
        convert(Ptr{Float64}, metadata_mapping.address),
        (METADATA_LENGTH,),
    )
    stream = PyRTCStream{T,N}(
        name,
        shape,
        payload_mapping,
        metadata_mapping,
        payload,
        metadata,
        owner,
        false,
        false,
        0,
        0,
        0.0,
    )
    finalizer(stream) do value
        _close_noexcept!(value)
    end
    return stream
end

function create_stream(
    name::AbstractString,
    ::Type{T},
    shape::NTuple{N,Int},
) where {T,N}
    Sys.islinux() || throw(PyRTCSharedMemoryError(
        :create,
        String(name),
        "the native pyRTC v1 adapter currently supports Linux only",
    ))
    stream_name = _validate_stream_name(name)
    payload_bytes = _checked_payload_bytes(stream_name, T, shape)
    pyrtc_dtype_code(T)
    payload_mapping = _create_mapping(stream_name, payload_bytes)
    metadata_mapping = nothing
    stream = nothing
    try
        metadata_mapping = _create_mapping(
            _metadata_name(stream_name),
            METADATA_BYTES,
        )
        stream = _stream_from_mappings(
            stream_name,
            T,
            shape,
            payload_mapping,
            metadata_mapping;
            owner=true,
        )
        metadata = stream.metadata
        fill!(metadata, 0.0)
        metadata[3] = Float64(payload_bytes)
        metadata[4] = Float64(pyrtc_dtype_code(T))
        for dimension_index in 1:N
            metadata[4 + dimension_index] = Float64(shape[dimension_index])
        end
        Base.Threads.atomic_fence()
        return stream
    catch
        if !isnothing(stream)
            _close_noexcept!(stream)
        else
            !isnothing(metadata_mapping) &&
                _close_mapping_noexcept!(metadata_mapping)
            _close_mapping_noexcept!(payload_mapping)
        end
        !isnothing(metadata_mapping) &&
            _unlink_name(_metadata_name(stream_name); missing_ok=true)
        _unlink_name(stream_name; missing_ok=true)
        rethrow()
    end
end

function open_stream(
    name::AbstractString,
    ::Type{T},
    shape::NTuple{N,Int},
) where {T,N}
    Sys.islinux() || throw(PyRTCSharedMemoryError(
        :open,
        String(name),
        "the native pyRTC v1 adapter currently supports Linux only",
    ))
    stream_name = _validate_stream_name(name)
    payload_bytes = _checked_payload_bytes(stream_name, T, shape)
    pyrtc_dtype_code(T)
    metadata_mapping = _open_mapping(
        _metadata_name(stream_name),
        METADATA_BYTES,
    )
    payload_mapping = nothing
    stream = nothing
    try
        metadata = UnsafeArray(
            convert(Ptr{Float64}, metadata_mapping.address),
            (METADATA_LENGTH,),
        )
        Base.Threads.atomic_fence()
        _validate_metadata!(
            metadata,
            stream_name,
            T,
            shape,
            payload_bytes,
        )
        payload_mapping = _open_mapping(stream_name, payload_bytes)
        stream = _stream_from_mappings(
            stream_name,
            T,
            shape,
            payload_mapping,
            metadata_mapping;
            owner=false,
        )
        return stream
    catch
        if !isnothing(stream)
            _close_noexcept!(stream)
        else
            !isnothing(payload_mapping) &&
                _close_mapping_noexcept!(payload_mapping)
            _close_mapping_noexcept!(metadata_mapping)
        end
        rethrow()
    end
end

function _copy_to_payload!(
    payload::UnsafeArray{T,1},
    source::AbstractVector{T},
) where {T}
    length(payload) == length(source) || throw(DimensionMismatch(
        "source length $(length(source)) does not match stream length $(length(payload))",
    ))
    output_index = 1
    @inbounds for source_index in eachindex(source)
        payload[output_index] = source[source_index]
        output_index += 1
    end
    return nothing
end

function _copy_to_payload!(
    payload::UnsafeArray{T,1},
    source::AbstractMatrix{T},
) where {T}
    length(payload) == length(source) || throw(DimensionMismatch(
        "source size $(size(source)) does not match stream element count $(length(payload))",
    ))
    output_index = 1
    @inbounds for row in axes(source, 1), column in axes(source, 2)
        payload[output_index] = source[row, column]
        output_index += 1
    end
    return nothing
end

function _copy_from_payload!(
    destination::AbstractVector{T},
    payload::UnsafeArray{T,1},
) where {T}
    length(destination) == length(payload) || throw(DimensionMismatch(
        "destination length $(length(destination)) does not match stream length $(length(payload))",
    ))
    input_index = 1
    @inbounds for destination_index in eachindex(destination)
        destination[destination_index] = payload[input_index]
        input_index += 1
    end
    return nothing
end

function _copy_from_payload!(
    destination::AbstractMatrix{T},
    payload::UnsafeArray{T,1},
) where {T}
    length(destination) == length(payload) || throw(DimensionMismatch(
        "destination size $(size(destination)) does not match stream element count $(length(payload))",
    ))
    input_index = 1
    @inbounds for row in axes(destination, 1), column in axes(destination, 2)
        destination[row, column] = payload[input_index]
        input_index += 1
    end
    return nothing
end

function publish!(
    stream::PyRTCStream{T,1},
    source::AbstractVector{T},
) where {T}
    _require_open(stream, :publish)
    size(source) == stream.shape || throw(DimensionMismatch(
        "source size $(size(source)) does not match stream shape $(stream.shape)",
    ))
    _copy_to_payload!(stream.payload, source)
    return _publish_metadata!(stream)
end

function publish!(
    stream::PyRTCStream{T,2},
    source::AbstractMatrix{T},
) where {T}
    _require_open(stream, :publish)
    size(source) == stream.shape || throw(DimensionMismatch(
        "source size $(size(source)) does not match stream shape $(stream.shape)",
    ))
    _copy_to_payload!(stream.payload, source)
    return _publish_metadata!(stream)
end

function _publish_metadata!(stream::PyRTCStream)
    stream.write_count < MAX_EXACT_FLOAT64_INTEGER || throw(
        PyRTCSharedMemoryError(
            :publish,
            stream.name,
            "write counter exhausted the pyRTC v1 exact-integer range",
        ),
    )
    stream.write_count += 1
    previous_timestamp = stream.metadata[2]
    write_timestamp = time()
    write_timestamp == previous_timestamp &&
        (write_timestamp = nextfloat(write_timestamp))
    Base.Threads.atomic_fence()
    stream.metadata[1] = Float64(stream.write_count)
    Base.Threads.atomic_fence()
    stream.metadata[2] = write_timestamp
    return stream.write_count
end

function stream_count(stream::PyRTCStream)
    _require_open(stream, :read)
    Base.Threads.atomic_fence()
    return _metadata_integer(
        stream.metadata[1],
        stream.name,
        "write count";
        allow_zero=true,
    )
end

@inline function _wait_budget_exhausted(start_time::UInt64, timeout_ns::UInt64)
    return time_ns() - start_time >= timeout_ns
end

function _timeout_nanoseconds(
    stream::PyRTCStream,
    timeout::Real,
)
    isfinite(timeout) && timeout > 0 || throw(PyRTCSharedMemoryError(
        :wait,
        stream.name,
        "timeout must be finite and positive",
    ))
    timeout <= typemax(UInt64) / 1.0e9 || throw(PyRTCSharedMemoryError(
        :wait,
        stream.name,
        "timeout is too large",
    ))
    return round(UInt64, timeout * 1.0e9)
end

function read_next!(
    destination::AbstractVector{T},
    stream::PyRTCStream{T,1};
    timeout::Real=5.0,
    poll_interval::Real=1.0e-5,
) where {T}
    size(destination) == stream.shape || throw(DimensionMismatch(
        "destination size $(size(destination)) does not match stream shape $(stream.shape)",
    ))
    return _read_next!(destination, stream, timeout, poll_interval)
end

function read_next!(
    destination::AbstractMatrix{T},
    stream::PyRTCStream{T,2};
    timeout::Real=5.0,
    poll_interval::Real=1.0e-5,
) where {T}
    size(destination) == stream.shape || throw(DimensionMismatch(
        "destination size $(size(destination)) does not match stream shape $(stream.shape)",
    ))
    return _read_next!(destination, stream, timeout, poll_interval)
end

function _read_next!(
    destination,
    stream::PyRTCStream,
    timeout::Real,
    poll_interval::Real,
)
    _require_open(stream, :read)
    isfinite(poll_interval) && poll_interval > 0 || throw(
        PyRTCSharedMemoryError(
            :wait,
            stream.name,
            "poll interval must be finite and positive",
        ),
    )
    timeout_ns = _timeout_nanoseconds(stream, timeout)
    start_time = time_ns()
    while true
        timestamp_before = stream.metadata[2]
        count_before = stream.metadata[1]
        if timestamp_before != 0.0 &&
                timestamp_before != stream.last_read_timestamp
            Base.Threads.atomic_fence()
            _copy_from_payload!(destination, stream.payload)
            Base.Threads.atomic_fence()
            count_after = stream.metadata[1]
            timestamp_after = stream.metadata[2]
            if count_before == count_after && timestamp_before == timestamp_after
                count = _metadata_integer(
                    count_after,
                    stream.name,
                    "write count";
                    allow_zero=false,
                )
                count > stream.last_read_count || throw(
                    PyRTCSharedMemoryError(
                        :read,
                        stream.name,
                        "write count $count did not advance beyond " *
                        "$(stream.last_read_count)",
                    ),
                )
                stream.last_read_count = count
                stream.last_read_timestamp = timestamp_after
                return count
            end
        end
        _wait_budget_exhausted(start_time, timeout_ns) && throw(
            PyRTCSharedMemoryError(
                :wait,
                stream.name,
                "no stable new publication arrived within $(Float64(timeout)) seconds",
            ),
        )
        sleep(poll_interval)
    end
end

function Base.close(stream::PyRTCStream)
    stream.closed && return nothing
    _close_mapping!(stream.payload_mapping)
    _close_mapping!(stream.metadata_mapping)
    stream.closed = true
    return nothing
end

function _close_noexcept!(stream::PyRTCStream)
    stream.closed && return nothing
    _close_mapping_noexcept!(stream.payload_mapping)
    _close_mapping_noexcept!(stream.metadata_mapping)
    stream.closed = true
    return nothing
end

function unlink!(stream::PyRTCStream)
    stream.owner || throw(PyRTCSharedMemoryError(
        :unlink,
        stream.name,
        "only the process that created the stream may unlink it",
    ))
    stream.unlinked && return nothing
    _unlink_name(_metadata_name(stream.name); missing_ok=true)
    _unlink_name(stream.name; missing_ok=true)
    stream.unlinked = true
    return nothing
end

end # module PyRTCSharedMemory

