#include "utils.cuh"
#include "macro.cuh"
#include "float.h"
#include "macro.cuh"

DEFINE_OP1_KERNEL(exp_kernel, expf)

DEFINE_OP1_KERNEL2(cuda_ln_kernel, float, log)
DEFINE_OP1_KERNEL2(cuda_ln_kernel, double, log)

DEFINE_OP1_KERNEL(sqrt_kernel, sqrtf)
DEFINE_OP1_KERNEL(sin_kernel, sin)
DEFINE_OP1_KERNEL(cos_kernel, cos)
DEFINE_OP1_KERNEL(tanh_kernel, tanh)

DEFINE_OP2_KERNEL3(cuda_pow_kernel, float, powf)
DEFINE_OP2_KERNEL3(cuda_pow_kernel, double, powf)

__global__ void sum_kernel(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    float sum = 0;
    for (size_t i = 0; i < len; i++) {
        sum += a[a_base_offset + i];
    }
    *b = sum;
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

__global__ void max_axis_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, NDimArray b_shape,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dim = a_shape.data[axis];

        float max = -FLT_MAX;
        for (size_t i = 0; i < dim; i++) {
            size_t a_idx = compute_offset_by_axis_index(a_base_offset, &b_shape.data[0], &a_strides.data[0], a_ndim, idx, axis, i);
            if (a[a_idx] > max) {
                max = a[a_idx];
            }
        }
        b[idx] = max;
    }
}

__global__ void argmax_axis_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, NDimArray b_shape,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dim = a_shape.data[axis];

        float max = -FLT_MAX;
        float max_idx = 0;
        for (size_t i = 0; i < dim; i++) {
            size_t a_idx = compute_offset_by_axis_index(a_base_offset, &b_shape.data[0], &a_strides.data[0], a_ndim, idx, axis, i);
            if (a[a_idx] > max) {
                max = a[a_idx];
                max_idx = (float)i;
            }
        }
        b[idx] = max_idx;
    }
}

extern "C" void cuda_sum(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    sum_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}

#define DEFINE_CUDA_SUM_AXIS(type) \
__global__ void cuda_sum_axis_kernel_##type( \
    type* a, Layout a_layout, type* b, Layout b_layout, size_t axis, int len \
) { \
    cuda_sum_axis_kernel<type>(a, a_layout, b, b_layout, axis, len); \
} \
extern "C" void cuda_sum_axis_##type( \
    type* a, Layout a_layout, type* b, Layout b_layout, size_t axis, int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    cuda_sum_axis_kernel_##type<<<blocks, threads>>>(a, a_layout, b, b_layout, axis, len ); \
}

DEFINE_CUDA_SUM_AXIS(float);
DEFINE_CUDA_SUM_AXIS(double);

extern "C" void cuda_max_axis(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t* b_shape,
    size_t axis,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, a_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    max_axis_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_shape_array,
        axis,
        len
    );
}

extern "C" void cuda_argmax_axis(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t* b_shape,
    size_t axis,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, a_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    argmax_axis_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_shape_array,
        axis,
        len
    );
}

extern "C" void cuda_exp(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    exp_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}

DEFINE_EXTERN_OP1_KERNEL(cuda_ln, float)
DEFINE_EXTERN_OP1_KERNEL(cuda_ln, double)

extern "C" void cuda_sqrt(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    sqrt_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}

DEFINE_EXTERN_OP2_KERNEL(cuda_pow, float)
DEFINE_EXTERN_OP2_KERNEL(cuda_pow, double)

extern "C" void cuda_sin(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    sin_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}

extern "C" void cuda_cos(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    cos_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}

extern "C" void cuda_tanh(
    float* a, size_t a_base_offset,
    float* b,
    int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    tanh_kernel<<<blocks, threads>>>(
        a, a_base_offset,
        b,
        len
    );
}
