#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>
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
