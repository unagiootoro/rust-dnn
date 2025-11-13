#include "utils.cuh"
#include "macro.cuh"
#include "float.h"
#include "macro.cuh"

DEFINE_OP1_KERNEL_FLOAT(cuda_exp, expf)
DEFINE_OP1_KERNEL_FLOAT(cuda_ln, log)
DEFINE_OP1_KERNEL_FLOAT(cuda_sqrt, sqrtf)
DEFINE_OP1_KERNEL_FLOAT(cuda_sin, sin)
DEFINE_OP1_KERNEL_FLOAT(cuda_cos, cos)
DEFINE_OP1_KERNEL_FLOAT(cuda_tanh, tanh)

DEFINE_OP2_KERNEL3_FLOAT(cuda_pow, powf)

template <typename T>
__device__ void cuda_sum_kernel(
    T* a, Layout a_layout,
    T* b,
    int len
) {
    T sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += a[a_layout.storage_offset + i];
    }
    *b = sum;
}

template <typename T>
__device__ void cuda_max_kernel(
    T* a, Layout a_layout,
    T* b,
    int len
) {
    T max = 0;
    for (size_t i = 0; i < len; i++) {
        T current = a[a_layout.storage_offset + i];
        if (current > max) {
            max = current;
        }
    }
    *b = max;
}

template <typename T>
__device__ void cuda_sum_axis_kernel(
    T* a, Layout a_layout,
    T* b, Layout b_layout,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dim = a_layout.shape[axis];

        T sum = 0;
        for (size_t i = 0; i < dim; i++) {
            size_t a_idx = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, i);
            sum += a[a_idx];
        }
        b[idx] = sum;
    }
}

template <typename T>
__device__ void cuda_max_axis_kernel(
    T* a, Layout a_layout,
    T* b, Layout b_layout,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dim = a_layout.shape[axis];

        T max = -FLT_MAX;
        for (size_t i = 0; i < dim; i++) {
            size_t a_idx = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, i);
            if (a[a_idx] > max) {
                max = a[a_idx];
            }
        }
        b[idx] = max;
    }
}

template <typename T>
__device__ void cuda_argmax_axis_kernel(
    T* a, Layout a_layout,
    uint32_t* b, Layout b_layout,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dim = a_layout.shape[axis];

        T max = -FLT_MAX;
        uint32_t max_idx = 0;
        for (size_t i = 0; i < dim; i++) {
            size_t a_idx = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, i);
            if (a[a_idx] > max) {
                max = a[a_idx];
                max_idx = (uint32_t)i;
            }
        }
        b[idx] = max_idx;
    }
}

#define DEFINE_REDUCE_TYPE(func_name, type) \
__global__ void func_name##_kernel_##type( \
    type* a, Layout a_layout, type* b, int len \
) { \
    func_name##_kernel<type>(a, a_layout, b, len); \
} \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, type* b, int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>(a, a_layout, b, len ); \
}

#define DEFINE_REDUCE_FLOAT(func_name) \
DEFINE_REDUCE_TYPE(func_name, float) \
DEFINE_REDUCE_TYPE(func_name, double)

DEFINE_REDUCE_FLOAT(cuda_sum)
DEFINE_REDUCE_FLOAT(cuda_max)

#define DEFINE_REDUCE_AXIS_TYPE(func_name, type) \
__global__ void func_name##_kernel_##type( \
    type* a, Layout a_layout, type* b, Layout b_layout, size_t axis, int len \
) { \
    func_name##_kernel<type>(a, a_layout, b, b_layout, axis, len); \
} \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, type* b, Layout b_layout, size_t axis, int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>(a, a_layout, b, b_layout, axis, len ); \
}

#define DEFINE_REDUCE_AXIS_FLOAT(func_name) \
DEFINE_REDUCE_AXIS_TYPE(func_name, float) \
DEFINE_REDUCE_AXIS_TYPE(func_name, double)

DEFINE_REDUCE_AXIS_FLOAT(cuda_sum_axis)
DEFINE_REDUCE_AXIS_FLOAT(cuda_max_axis)

#define DEFINE_REDUCE_CMP_AXIS_INDEX_TYPE(func_name, type) \
__global__ void func_name##_kernel_##type( \
    type* a, Layout a_layout, uint32_t* b, Layout b_layout, size_t axis, int len \
) { \
    func_name##_kernel<type>(a, a_layout, b, b_layout, axis, len); \
} \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, uint32_t* b, Layout b_layout, size_t axis, int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>(a, a_layout, b, b_layout, axis, len ); \
}

#define DEFINE_REDUCE_CMP_AXIS_INDEX_FLOAT(func_name) \
DEFINE_REDUCE_CMP_AXIS_INDEX_TYPE(func_name, float) \
DEFINE_REDUCE_CMP_AXIS_INDEX_TYPE(func_name, double)

DEFINE_REDUCE_CMP_AXIS_INDEX_FLOAT(cuda_argmax_axis)
