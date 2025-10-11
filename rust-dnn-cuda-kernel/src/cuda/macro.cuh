#pragma once

#include "cstdint"
#include "common.cuh"

#define DEFINE_OP1_KERNEL(func_name, op) \
__global__ void func_name( \
    float* a, size_t a_base_offset, \
    float* b, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        b[idx] = op(a[a_base_offset + idx]); \
    } \
}

#define DEFINE_OP1_KERNEL2(func_name, type, op) \
__global__ void func_name##_##type( \
    type* a, size_t a_base_offset, \
    type* b, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        b[idx] = op(a[a_base_offset + idx]); \
    } \
}

#define DEFINE_OP2_KERNEL(func_name, op) \
__global__ void func_name( \
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim, \
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim, \
    float* c, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx); \
        size_t b_idx = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx); \
        c[idx] = a[a_idx] op b[b_idx]; \
    } \
}

#define DEFINE_OP2_KERNEL2(func_name, type, op) \
__global__ void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    type* c, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset2(&a_layout, idx); \
        size_t b_idx = compute_offset2(&b_layout, idx); \
        c[idx] = a[a_idx] op b[b_idx]; \
    } \
}

#define DEFINE_OP2_KERNEL3(func_name, type, op) \
__global__ void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    type* c, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset2(&a_layout, idx); \
        size_t b_idx = compute_offset2(&b_layout, idx); \
        c[idx] = op(a[a_idx], b[b_idx]); \
    } \
}

#define DEFINE_OP2_ASSIGN_KERNEL(func_name, op) \
__global__ void func_name( \
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim, \
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx); \
        size_t b_idx = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx); \
        a[a_idx] op b[b_idx]; \
    } \
}

#define DEFINE_CMP_OP2_KERNEL(func_name, op) \
__global__ void func_name( \
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim, \
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim, \
    float* c, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx); \
        size_t b_idx = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx); \
        if (a[a_idx] op b[b_idx]) { \
            c[idx] = 1.0; \
        } else { \
            c[idx] = 0.0; \
        } \
    } \
}

#define DEFINE_EXTERN_OP1_KERNEL(func_name, type) \
extern "C" void func_name##_##type( \
    type* a, size_t a_base_offset, \
    type* b, \
    int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>( \
        a, a_base_offset, \
        b, \
        len \
    ); \
}

#define DEFINE_EXTERN_OP2_KERNEL(func_name, type) \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    type* c, \
    int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>( \
        a, a_layout, \
        b, b_layout, \
        c, \
        len \
    ); \
}
