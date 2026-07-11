#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void add_one_i32(std::int32_t* data, std::size_t len) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < len) {
        data[index] += 1;
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

