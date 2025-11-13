#pragma once

#include "cstdint"
#include "common.cuh"

#define DEFINE_OP1_KERNEL_DEVICE(func_name, type, op) \
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

#define DEFINE_OP2_KERNEL_DEVICE(func_name, type, op) \
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

#define DEFINE_OP2_U32_KERNEL_DEVICE(func_name, type, op) \
__global__ void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    uint32_t* c, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset2(&a_layout, idx); \
        size_t b_idx = compute_offset2(&b_layout, idx); \
        c[idx] = a[a_idx] op b[b_idx]; \
    } \
}

#define DEFINE_OP2_KERNEL_DEVICE2(func_name, type, op) \
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

#define DEFINE_OP2_ASSIGN_KERNEL_DEVICE(func_name, type, op) \
__global__ void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    int len \
) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < len) { \
        size_t a_idx = compute_offset2(&a_layout, idx); \
        size_t b_idx = compute_offset2(&b_layout, idx); \
        a[a_idx] op b[b_idx]; \
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

#define DEFINE_EXTERN_OP2_U32_KERNEL(func_name, type) \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    uint32_t* c, \
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

#define DEFINE_EXTERN_OP2_ASSIGN_KERNEL(func_name, type) \
extern "C" void func_name##_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    func_name##_kernel_##type<<<blocks, threads>>>( \
        a, a_layout, \
        b, b_layout, \
        len \
    ); \
}

#define DEFINE_OP1_KERNEL_TYPE(func_name, type, op) \
DEFINE_OP1_KERNEL_DEVICE(func_name##_kernel, type, op) \
DEFINE_EXTERN_OP1_KERNEL(func_name, type)

#define DEFINE_OP1_KERNEL2(func_name, op) \
DEFINE_OP1_KERNEL_TYPE(func_name, uint32_t, op) \
DEFINE_OP1_KERNEL_TYPE(func_name, float, op) \
DEFINE_OP1_KERNEL_TYPE(func_name, double, op)

#define DEFINE_OP1_KERNEL_FLOAT(func_name, op) \
DEFINE_OP1_KERNEL_TYPE(func_name, float, op) \
DEFINE_OP1_KERNEL_TYPE(func_name, double, op)

#define DEFINE_OP2_KERNEL_TYPE(func_name, type, op) \
DEFINE_OP2_KERNEL_DEVICE(func_name##_kernel, type, op) \
DEFINE_EXTERN_OP2_KERNEL(func_name, type)

#define DEFINE_OP2_KERNEL2(func_name, op) \
DEFINE_OP2_KERNEL_TYPE(func_name, uint32_t, op) \
DEFINE_OP2_KERNEL_TYPE(func_name, float, op) \
DEFINE_OP2_KERNEL_TYPE(func_name, double, op)

#define DEFINE_OP2_U32_KERNEL_TYPE(func_name, type, op) \
DEFINE_OP2_U32_KERNEL_DEVICE(func_name##_kernel, type, op) \
DEFINE_EXTERN_OP2_U32_KERNEL(func_name, type)

#define DEFINE_OP2_U32_KERNEL2(func_name, op) \
DEFINE_OP2_U32_KERNEL_TYPE(func_name, uint32_t, op) \
DEFINE_OP2_U32_KERNEL_TYPE(func_name, float, op) \
DEFINE_OP2_U32_KERNEL_TYPE(func_name, double, op)

#define DEFINE_OP2_KERNEL2_TYPE(func_name, type, op) \
DEFINE_OP2_KERNEL_DEVICE2(func_name##_kernel, type, op) \
DEFINE_EXTERN_OP2_KERNEL(func_name, type)

#define DEFINE_OP2_ASSIGN_KERNEL2_TYPE(func_name, type, op) \
DEFINE_OP2_ASSIGN_KERNEL_DEVICE(func_name##_kernel, type, op) \
DEFINE_EXTERN_OP2_ASSIGN_KERNEL(func_name, type)

#define DEFINE_OP2_ASSIGN_KERNEL2(func_name, op) \
DEFINE_OP2_ASSIGN_KERNEL2_TYPE(func_name, uint32_t, op) \
DEFINE_OP2_ASSIGN_KERNEL2_TYPE(func_name, float, op) \
DEFINE_OP2_ASSIGN_KERNEL2_TYPE(func_name, double, op)

#define DEFINE_OP2_KERNEL3_FLOAT(func_name, op) \
DEFINE_OP2_KERNEL2_TYPE(func_name, float, op) \
DEFINE_OP2_KERNEL2_TYPE(func_name, double, op)
