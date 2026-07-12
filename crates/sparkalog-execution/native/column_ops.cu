#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void add_one_i32(std::int32_t* data, std::size_t len) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < len) {
        data[index] += 1;
    }
}

__device__ bool compare_u32(std::uint32_t value, std::uint32_t operation, std::uint32_t operand) {
    switch (operation) {
        case 0: return value == operand;
        case 1: return value != operand;
        case 2: return value < operand;
        case 3: return value <= operand;
        case 4: return value > operand;
        case 5: return value >= operand;
        default: return false;
    }
}

__global__ void mark_filter_u32(
    const std::uint32_t* input,
    std::size_t len,
    std::uint32_t operation,
    std::uint32_t operand,
    std::uint32_t* flags) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto index = start; index < len; index += stride) {
        flags[index] = compare_u32(input[index], operation, operand) ? 1U : 0U;
    }
}

__global__ void fill_mod_u32(std::uint32_t* output, std::size_t len, std::uint32_t modulus) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto index = start; index < len; index += stride) {
        output[index] = static_cast<std::uint32_t>(index % modulus);
    }
}

__device__ std::int64_t find_index_key(
    const std::uint32_t* keys,
    std::size_t unique_keys,
    std::uint32_t key) {
    std::size_t left = 0;
    std::size_t right = unique_keys;
    while (left < right) {
        const auto middle = left + (right - left) / 2;
        const auto candidate = keys[middle];
        if (candidate < key) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    if (left < unique_keys && keys[left] == key) {
        return static_cast<std::int64_t>(left);
    }
    return -1;
}

__global__ void count_join_u32(
    const std::uint32_t* left_keys,
    std::size_t left_rows,
    const std::uint32_t* index_keys,
    const std::uint32_t* index_starts,
    std::size_t unique_keys,
    std::uint64_t* counts) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto row = start; row < left_rows; row += stride) {
        const auto index = find_index_key(index_keys, unique_keys, left_keys[row]);
        counts[row] = index < 0
            ? 0
            : static_cast<std::uint64_t>(index_starts[index + 1] - index_starts[index]);
    }
}

__global__ void finish_join_count(
    const std::uint64_t* counts,
    const std::uint64_t* offsets,
    std::size_t left_rows,
    std::uint64_t* total) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *total = left_rows == 0 ? 0 : offsets[left_rows - 1] + counts[left_rows - 1];
    }
}

__global__ void emit_join_u32(
    const std::uint32_t* left_keys,
    std::size_t left_rows,
    const std::uint32_t* index_keys,
    const std::uint32_t* index_starts,
    const std::uint32_t* index_rows,
    std::size_t unique_keys,
    const std::uint32_t* projection0,
    std::uint32_t projection0_side,
    const std::uint32_t* projection1,
    std::uint32_t projection1_side,
    const std::uint64_t* offsets,
    std::uint32_t* output0,
    std::uint32_t* output1) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto left_row = start; left_row < left_rows; left_row += stride) {
        const auto index = find_index_key(index_keys, unique_keys, left_keys[left_row]);
        if (index < 0) {
            continue;
        }
        const auto match_start = index_starts[index];
        const auto match_end = index_starts[index + 1];
        for (auto match = match_start; match < match_end; ++match) {
            const auto right_row = index_rows[match];
            const auto output_row = offsets[left_row] + (match - match_start);
            output0[output_row] = projection0_side == 0
                ? projection0[left_row]
                : projection0[right_row];
            output1[output_row] = projection1_side == 0
                ? projection1[left_row]
                : projection1[right_row];
        }
    }
}

__global__ void pack_binary_u32(
    const std::uint32_t* first,
    const std::uint32_t* second,
    std::size_t rows,
    std::uint64_t* packed) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto row = start; row < rows; row += stride) {
        packed[row] = (static_cast<std::uint64_t>(first[row]) << 32) | second[row];
    }
}

__global__ void unpack_binary_u32(
    const std::uint64_t* packed,
    const std::uint64_t* unique_rows,
    std::size_t input_rows,
    std::uint32_t* first,
    std::uint32_t* second) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const auto rows = *unique_rows;
    for (auto row = start; row < input_rows; row += stride) {
        if (row < rows) {
            const auto tuple = packed[row];
            first[row] = static_cast<std::uint32_t>(tuple >> 32);
            second[row] = static_cast<std::uint32_t>(tuple);
        }
    }
}

__device__ bool contains_binary_tuple(
    const std::uint32_t* first,
    const std::uint32_t* second,
    std::size_t rows,
    std::uint32_t target_first,
    std::uint32_t target_second) {
    std::size_t low = 0;
    std::size_t high = rows;
    while (low < high) {
        const auto middle = low + (high - low) / 2;
        const auto candidate_first = first[middle];
        const auto candidate_second = second[middle];
        if (candidate_first < target_first ||
            (candidate_first == target_first && candidate_second < target_second)) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < rows && first[low] == target_first && second[low] == target_second;
}

__global__ void mark_sorted_binary_anti_join(
    const std::uint32_t* left_first,
    const std::uint32_t* left_second,
    std::size_t left_rows,
    const std::uint32_t* right_first,
    const std::uint32_t* right_second,
    std::size_t right_rows,
    std::uint32_t* flags) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (auto row = start; row < left_rows; row += stride) {
        flags[row] = contains_binary_tuple(
            right_first,
            right_second,
            right_rows,
            left_first[row],
            left_second[row]) ? 0U : 1U;
    }
}

__global__ void gather_binary_rows(
    const std::uint32_t* input_first,
    const std::uint32_t* input_second,
    const std::uint32_t* selected,
    const std::uint32_t* selected_rows,
    std::size_t input_rows,
    std::uint32_t* output_first,
    std::uint32_t* output_second) {
    const auto start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    const auto rows = *selected_rows;
    for (auto output_row = start; output_row < input_rows; output_row += stride) {
        if (output_row < rows) {
            const auto input_row = selected[output_row];
            output_first[output_row] = input_first[input_row];
            output_second[output_row] = input_second[input_row];
        }
    }
}

}  // namespace

extern "C" cudaError_t sparkalog_add_one_i32(
    std::int32_t* data,
    std::size_t len,
    void* stream) {
    if (data == nullptr && len != 0) {
        return cudaErrorInvalidValue;
    }
    if (len == 0) {
        return cudaSuccess;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((len + threads - 1) / threads);
    add_one_i32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(data, len);
    return cudaGetLastError();
}

extern "C" cudaError_t sparkalog_stream_synchronize(void* stream) {
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

extern "C" cudaError_t sparkalog_filter_u32_temporary_bytes(
    const std::uint32_t* flags,
    std::uint32_t* output,
    std::uint32_t* count,
    std::size_t len,
    std::size_t* temporary_bytes,
    void* stream) {
    if (temporary_bytes == nullptr || count == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (len != 0 && (flags == nullptr || output == nullptr)) {
        return cudaErrorInvalidValue;
    }

    auto input = thrust::make_counting_iterator<std::uint32_t>(0);
    return cub::DeviceSelect::Flagged(
        nullptr,
        *temporary_bytes,
        input,
        flags,
        output,
        count,
        static_cast<::cuda::std::int64_t>(len),
        static_cast<cudaStream_t>(stream));
}

extern "C" cudaError_t sparkalog_filter_u32(
    const std::uint32_t* input,
    std::size_t len,
    std::uint32_t operation,
    std::uint32_t operand,
    std::uint32_t* flags,
    std::uint32_t* output,
    std::uint32_t* count,
    void* temporary,
    std::size_t temporary_bytes,
    void* stream) {
    if (count == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (len == 0) {
        return cudaMemsetAsync(count, 0, sizeof(*count), static_cast<cudaStream_t>(stream));
    }
    if (input == nullptr || flags == nullptr || output == nullptr || temporary == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((len + threads - 1) / threads);
    mark_filter_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        input, len, operation, operand, flags);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }

    auto row_ids = thrust::make_counting_iterator<std::uint32_t>(0);
    return cub::DeviceSelect::Flagged(
        temporary,
        temporary_bytes,
        row_ids,
        flags,
        output,
        count,
        static_cast<::cuda::std::int64_t>(len),
        static_cast<cudaStream_t>(stream));
}

extern "C" cudaError_t sparkalog_fill_mod_u32(
    std::uint32_t* output,
    std::size_t len,
    std::uint32_t modulus,
    void* stream) {
    if (modulus == 0 || (output == nullptr && len != 0)) {
        return cudaErrorInvalidValue;
    }
    if (len == 0) {
        return cudaSuccess;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((len + threads - 1) / threads);
    fill_mod_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        output, len, modulus);
    return cudaGetLastError();
}

extern "C" cudaError_t sparkalog_join_u32_temporary_bytes(
    const std::uint64_t* counts,
    std::uint64_t* offsets,
    std::size_t left_rows,
    std::size_t* temporary_bytes,
    void* stream) {
    if (temporary_bytes == nullptr || (left_rows != 0 && (counts == nullptr || offsets == nullptr))) {
        return cudaErrorInvalidValue;
    }
    return cub::DeviceScan::ExclusiveSum(
        nullptr,
        *temporary_bytes,
        counts,
        offsets,
        static_cast<::cuda::std::int64_t>(left_rows),
        static_cast<cudaStream_t>(stream));
}

extern "C" cudaError_t sparkalog_join_u32_count(
    const std::uint32_t* left_keys,
    std::size_t left_rows,
    const std::uint32_t* index_keys,
    const std::uint32_t* index_starts,
    std::size_t unique_keys,
    std::uint64_t* counts,
    std::uint64_t* offsets,
    std::uint64_t* total,
    void* temporary,
    std::size_t temporary_bytes,
    void* stream) {
    if (total == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (left_rows == 0) {
        return cudaMemsetAsync(total, 0, sizeof(*total), static_cast<cudaStream_t>(stream));
    }
    if (left_keys == nullptr || index_keys == nullptr || index_starts == nullptr ||
        counts == nullptr || offsets == nullptr || temporary == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((left_rows + threads - 1) / threads);
    count_join_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        left_keys, left_rows, index_keys, index_starts, unique_keys, counts);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    status = cub::DeviceScan::ExclusiveSum(
        temporary,
        temporary_bytes,
        counts,
        offsets,
        static_cast<::cuda::std::int64_t>(left_rows),
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    finish_join_count<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        counts, offsets, left_rows, total);
    return cudaGetLastError();
}

extern "C" cudaError_t sparkalog_join_u32_emit(
    const std::uint32_t* left_keys,
    std::size_t left_rows,
    const std::uint32_t* index_keys,
    const std::uint32_t* index_starts,
    const std::uint32_t* index_rows,
    std::size_t unique_keys,
    const std::uint32_t* projection0,
    std::uint32_t projection0_side,
    const std::uint32_t* projection1,
    std::uint32_t projection1_side,
    const std::uint64_t* offsets,
    std::uint32_t* output0,
    std::uint32_t* output1,
    void* stream) {
    if (left_rows == 0) {
        return cudaSuccess;
    }
    if (left_keys == nullptr || index_keys == nullptr || index_starts == nullptr ||
        index_rows == nullptr || projection0 == nullptr || projection1 == nullptr ||
        offsets == nullptr || output0 == nullptr || output1 == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((left_rows + threads - 1) / threads);
    emit_join_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        left_keys,
        left_rows,
        index_keys,
        index_starts,
        index_rows,
        unique_keys,
        projection0,
        projection0_side,
        projection1,
        projection1_side,
        offsets,
        output0,
        output1);
    return cudaGetLastError();
}

extern "C" cudaError_t sparkalog_distinct_u32_temporary_bytes(
    std::uint64_t* packed,
    std::uint64_t* scratch,
    std::uint64_t* unique_rows,
    std::size_t rows,
    std::size_t* temporary_bytes,
    void* stream) {
    if (temporary_bytes == nullptr || unique_rows == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (rows != 0 && (packed == nullptr || scratch == nullptr)) {
        return cudaErrorInvalidValue;
    }

    std::size_t sort_bytes = 0;
    auto status = cub::DeviceRadixSort::SortKeys(
        nullptr,
        sort_bytes,
        packed,
        scratch,
        static_cast<::cuda::std::int64_t>(rows),
        0,
        64,
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    std::size_t unique_bytes = 0;
    status = cub::DeviceSelect::Unique(
        nullptr,
        unique_bytes,
        scratch,
        packed,
        unique_rows,
        static_cast<::cuda::std::int64_t>(rows),
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    *temporary_bytes = sort_bytes > unique_bytes ? sort_bytes : unique_bytes;
    return cudaSuccess;
}

extern "C" cudaError_t sparkalog_distinct_u32(
    const std::uint32_t* first,
    const std::uint32_t* second,
    std::size_t rows,
    std::uint64_t* packed,
    std::uint64_t* scratch,
    std::uint64_t* unique_rows,
    std::uint32_t* output_first,
    std::uint32_t* output_second,
    void* temporary,
    std::size_t temporary_bytes,
    void* stream) {
    if (unique_rows == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (rows == 0) {
        return cudaMemsetAsync(
            unique_rows, 0, sizeof(*unique_rows), static_cast<cudaStream_t>(stream));
    }
    if (first == nullptr || second == nullptr || packed == nullptr || scratch == nullptr ||
        output_first == nullptr || output_second == nullptr || temporary == nullptr) {
        return cudaErrorInvalidValue;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((rows + threads - 1) / threads);
    pack_binary_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        first, second, rows, packed);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    std::size_t available = temporary_bytes;
    status = cub::DeviceRadixSort::SortKeys(
        temporary,
        available,
        packed,
        scratch,
        static_cast<::cuda::std::int64_t>(rows),
        0,
        64,
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    available = temporary_bytes;
    status = cub::DeviceSelect::Unique(
        temporary,
        available,
        scratch,
        packed,
        unique_rows,
        static_cast<::cuda::std::int64_t>(rows),
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    unpack_binary_u32<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        packed, unique_rows, rows, output_first, output_second);
    return cudaGetLastError();
}

extern "C" cudaError_t sparkalog_anti_join_u32_temporary_bytes(
    const std::uint32_t* flags,
    std::uint32_t* selected,
    std::uint32_t* selected_rows,
    std::size_t left_rows,
    std::size_t* temporary_bytes,
    void* stream) {
    if (temporary_bytes == nullptr || selected_rows == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (left_rows != 0 && (flags == nullptr || selected == nullptr)) {
        return cudaErrorInvalidValue;
    }
    auto row_ids = thrust::make_counting_iterator<std::uint32_t>(0);
    return cub::DeviceSelect::Flagged(
        nullptr,
        *temporary_bytes,
        row_ids,
        flags,
        selected,
        selected_rows,
        static_cast<::cuda::std::int64_t>(left_rows),
        static_cast<cudaStream_t>(stream));
}

extern "C" cudaError_t sparkalog_anti_join_u32(
    const std::uint32_t* left_first,
    const std::uint32_t* left_second,
    std::size_t left_rows,
    const std::uint32_t* right_first,
    const std::uint32_t* right_second,
    std::size_t right_rows,
    std::uint32_t* flags,
    std::uint32_t* selected,
    std::uint32_t* selected_rows,
    std::uint32_t* output_first,
    std::uint32_t* output_second,
    void* temporary,
    std::size_t temporary_bytes,
    void* stream) {
    if (selected_rows == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (left_rows == 0) {
        return cudaMemsetAsync(
            selected_rows, 0, sizeof(*selected_rows), static_cast<cudaStream_t>(stream));
    }
    if (left_first == nullptr || left_second == nullptr || flags == nullptr ||
        selected == nullptr || output_first == nullptr || output_second == nullptr ||
        temporary == nullptr ||
        (right_rows != 0 && (right_first == nullptr || right_second == nullptr))) {
        return cudaErrorInvalidValue;
    }

    constexpr unsigned int threads = 256;
    const auto blocks = static_cast<unsigned int>((left_rows + threads - 1) / threads);
    mark_sorted_binary_anti_join<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        left_first,
        left_second,
        left_rows,
        right_first,
        right_second,
        right_rows,
        flags);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    auto row_ids = thrust::make_counting_iterator<std::uint32_t>(0);
    status = cub::DeviceSelect::Flagged(
        temporary,
        temporary_bytes,
        row_ids,
        flags,
        selected,
        selected_rows,
        static_cast<::cuda::std::int64_t>(left_rows),
        static_cast<cudaStream_t>(stream));
    if (status != cudaSuccess) {
        return status;
    }
    gather_binary_rows<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        left_first,
        left_second,
        selected,
        selected_rows,
        left_rows,
        output_first,
        output_second);
    return cudaGetLastError();
}
