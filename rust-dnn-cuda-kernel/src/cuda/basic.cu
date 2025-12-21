#include "utils.cuh"
#include "macro.cuh"
#include "common.cuh"

DEFINE_OP1_KERNEL_FLOAT(cuda_neg, -)

DEFINE_OP2_KERNEL2(cuda_add, +)
DEFINE_OP2_KERNEL2(cuda_sub, -)
DEFINE_OP2_KERNEL2(cuda_mul, *)
DEFINE_OP2_KERNEL2(cuda_div, /)

DEFINE_OP2_ASSIGN_KERNEL2(cuda_copy, =)
DEFINE_OP2_ASSIGN_KERNEL2(cuda_add_assign, +=)
DEFINE_OP2_ASSIGN_KERNEL2(cuda_sub_assign, -=)
DEFINE_OP2_ASSIGN_KERNEL2(cuda_mul_assign, *=)
DEFINE_OP2_ASSIGN_KERNEL2(cuda_div_assign, /=)

DEFINE_OP2_U32_KERNEL2(cuda_lt, <)
DEFINE_OP2_U32_KERNEL2(cuda_le, <=)
DEFINE_OP2_U32_KERNEL2(cuda_gt, >)
DEFINE_OP2_U32_KERNEL2(cuda_ge, >=)
DEFINE_OP2_U32_KERNEL2(cuda_eq, ==)

template <typename T1, typename T2>
__device__ void cuda_convert_kernel(
    T1* a,
    Layout a_layout,
    T2* b,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_idx = compute_offset2(&a_layout, idx);
        b[idx] = a[a_idx];
    }
}

template <typename T>
__device__ void cuda_matmul_kernel(
    T* a, Layout a_layout,
    T* b, Layout b_layout,
    T* c,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_cols = a_layout.shape[1];
        size_t b_cols = b_layout.shape[1];

        size_t i = idx / b_cols;
        size_t j = idx % b_cols;
        T sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            size_t a_idx = compute_offset_by_indices_dim2(&a_layout, i, k);
            size_t b_idx = compute_offset_by_indices_dim2(&b_layout, k, j);
            sum += a[a_idx] * b[b_idx];
        }
        c[i * b_cols + j] = sum;
    }
}

template <typename T>
__device__ void cuda_contiguous_kernel(
    T* a,
    Layout a_layout,
    T* b,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_idx = compute_offset2(&a_layout, idx);
        b[idx] = a[a_idx];
    }
}

template <typename T>
__device__ void cuda_gather_kernel(
    T* a, Layout a_layout,
    uint32_t* b, Layout b_layout,
    T* c,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t b_index = compute_offset2(&b_layout, idx);
        size_t b_value = b[b_index];
        size_t a_index = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, b_value);
        c[idx] = a[a_index];
    }
}

template <typename T>
__device__ void cuda_scatter_kernel(
    T* a, Layout a_layout,
    uint32_t* b, Layout b_layout,
    T* c, Layout c_layout,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = b_layout.shape[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t b_index = compute_offset2(&b_layout, idx);
            size_t b_value = b[b_index];
            size_t a_index = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, b_value);
            size_t c_index = compute_offset2(&c_layout, idx);
            a[a_index] = c[c_index];
        }
    }
}

template <typename T>
__device__ void cuda_scatter_add_kernel(
    T* a, Layout a_layout,
    uint32_t* b, Layout b_layout,
    T* c, Layout c_layout,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = b_layout.shape[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t b_index = compute_offset2(&b_layout, idx);
            size_t b_value = b[b_index];
            size_t a_index = compute_offset_by_axis_index(a_layout.storage_offset, &b_layout.shape[0], &a_layout.stride[0], b_layout.ndim, idx, axis, b_value);
            size_t c_index = compute_offset2(&c_layout, idx);
            a[a_index] += c[c_index];
        }
    }
}

template <typename T>
__device__ void cuda_index_select_kernel(
    T* input_data, Layout input_layout,
    uint32_t* index_data, Layout index_layout,
    T* output_data,
    Layout output_layout,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t output_axis_index = unravel_index_axis(&output_layout.shape[0], output_layout.ndim, axis, idx);
        size_t index_offset = compute_offset2(&index_layout, output_axis_index);
        size_t index = index_data[index_offset];
        size_t input_offset = compute_offset_by_axis_index(input_layout.storage_offset, &output_layout.shape[0], &input_layout.stride[0], output_layout.ndim, idx, axis, index);
        output_data[idx] = input_data[input_offset];
    }
}

template <typename T>
__device__ void cuda_index_copy_kernel(
    T* input_data, Layout input_layout,
    uint32_t* index_data, Layout index_layout,
    T* src_data,
    Layout src_layout,
    NDimArray dest_shape,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = dest_shape.data[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t output_axis_index = unravel_index_axis(&dest_shape.data[0], dest_shape.ndim, axis, idx);
            size_t index_offset = compute_offset2(&index_layout, output_axis_index);
            size_t index = index_data[index_offset];
            size_t input_offset = compute_offset_by_axis_index(input_layout.storage_offset, &dest_shape.data[0], &input_layout.stride[0], dest_shape.ndim, idx, axis, index);
            size_t src_offset = compute_offset2(&src_layout, output_axis_index);
            input_data[input_offset] = src_data[src_offset];
        }
    }
}

template <typename T>
__device__ void cuda_index_add_kernel(
    T* input_data, Layout input_layout,
    uint32_t* index_data, Layout index_layout,
    T* src_data,
    Layout src_layout,
    NDimArray dest_shape,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = dest_shape.data[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t output_axis_index = unravel_index_axis(&dest_shape.data[0], dest_shape.ndim, axis, idx);
            size_t index_offset = compute_offset2(&index_layout, output_axis_index);
            size_t index = index_data[index_offset];
            size_t input_offset = compute_offset_by_axis_index(input_layout.storage_offset, &dest_shape.data[0], &input_layout.stride[0], dest_shape.ndim, idx, axis, index);
            size_t src_offset = compute_offset2(&src_layout, output_axis_index);
            input_data[input_offset] += src_data[src_offset];
        }
    }
}

template <typename T>
__device__ void cuda_fill_kernel(T* output_data, T value, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output_data[idx] = value;
    }
}

#define DEFINE_CUDA_CONVERT_KERNEL(type1, type2) \
__global__ void cuda_convert_kernel_##type1##_##type2( \
    type1* a, Layout a_layout, type2* b, int len \
) { \
    cuda_convert_kernel<type1, type2>(a, a_layout, b, len); \
}

DEFINE_CUDA_CONVERT_KERNEL(uint32_t, float)
DEFINE_CUDA_CONVERT_KERNEL(uint32_t, double)
DEFINE_CUDA_CONVERT_KERNEL(float, uint32_t)
DEFINE_CUDA_CONVERT_KERNEL(float, double)
DEFINE_CUDA_CONVERT_KERNEL(double, uint32_t)
DEFINE_CUDA_CONVERT_KERNEL(double, float)

extern "C" void cuda_convert(
    DType dtype1, DType dtype2, void* a, Layout a_layout, void* b, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype1) {
        case DType_U32:
            switch (dtype2) {
                case DType_F32:
                    cuda_convert_kernel_uint32_t_float<<<blocks, threads>>>((uint32_t*)a, a_layout, (float*)b, len);
                    break;
                case DType_F64:
                    cuda_convert_kernel_uint32_t_double<<<blocks, threads>>>((uint32_t*)a, a_layout, (double*)b, len);
                    break;
            }
            break;
        case DType_F32:
            switch (dtype2) {
                case DType_U32:
                    cuda_convert_kernel_float_uint32_t<<<blocks, threads>>>((float*)a, a_layout, (uint32_t*)b, len);
                    break;
                case DType_F64:
                    cuda_convert_kernel_float_double<<<blocks, threads>>>((float*)a, a_layout, (double*)b, len);
                    break;
            }
            break;
        case DType_F64:
            switch (dtype2) {
                case DType_U32:
                    cuda_convert_kernel_double_uint32_t<<<blocks, threads>>>((double*)a, a_layout, (uint32_t*)b, len);
                    break;
                case DType_F32:
                    cuda_convert_kernel_double_float<<<blocks, threads>>>((double*)a, a_layout, (float*)b, len);
                    break;
            }
            break;
    }
}

#define DEFINE_CUDA_MATMUL_KERNEL(type) \
__global__ void cuda_matmul_kernel_##type( \
    type* a, Layout a_layout, \
    type* b, Layout b_layout, \
    type* c, \
    int len \
) { \
    cuda_matmul_kernel<type>(a, a_layout, b, b_layout, c, len); \
}

DEFINE_CUDA_MATMUL_KERNEL(float)
DEFINE_CUDA_MATMUL_KERNEL(double)

extern "C" void cuda_matmul(
    DType dtype, void* a, Layout a_layout, uint32_t* b, Layout b_layout, void* c, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_F32:
            cuda_matmul_kernel_float<<<blocks, threads>>>((float*)a, a_layout, (float*)b, b_layout, (float*)c, len);
            break;
        case DType_F64:
            cuda_matmul_kernel_double<<<blocks, threads>>>((double*)a, a_layout, (double*)b, b_layout, (double*)c, len);
            break;
    }
}

#define DEFINE_CUDA_KERNEL_CONTIGUOUS(type) \
__global__ void cuda_contiguous_kernel_##type( \
    type* a, Layout a_layout, type* b, int len \
) { \
    cuda_contiguous_kernel<type>(a, a_layout, b, len); \
}

DEFINE_CUDA_KERNEL_CONTIGUOUS(uint32_t)
DEFINE_CUDA_KERNEL_CONTIGUOUS(float)
DEFINE_CUDA_KERNEL_CONTIGUOUS(double)

extern "C" void cuda_contiguous(
    DType dtype, void* a, Layout a_layout, void* b, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_contiguous_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)a, a_layout, (uint32_t*)b, len);
            break;
        case DType_F32:
            cuda_contiguous_kernel_float<<<blocks, threads>>>((float*)a, a_layout, (float*)b, len);
            break;
        case DType_F64:
            cuda_contiguous_kernel_double<<<blocks, threads>>>((double*)a, a_layout, (double*)b, len);
            break;
    }
}

#define DEFINE_CUDA_GATHER_KERNEL(type) \
__global__ void cuda_gather_kernel_##type( \
    type* a, Layout a_layout, uint32_t* b, Layout b_layout, type* c, size_t axis, int len \
) { \
    cuda_gather_kernel<type>(a, a_layout, b, b_layout, c, axis, len); \
}

DEFINE_CUDA_GATHER_KERNEL(uint32_t)
DEFINE_CUDA_GATHER_KERNEL(float)
DEFINE_CUDA_GATHER_KERNEL(double)

extern "C" void cuda_gather(
    DType dtype, void* a, Layout a_layout, uint32_t* b, Layout b_layout, void* c, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_gather_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)a, a_layout, b, b_layout, (uint32_t*)c, axis, len);
            break;
        case DType_F32:
            cuda_gather_kernel_float<<<blocks, threads>>>((float*)a, a_layout, b, b_layout, (float*)c, axis, len);
            break;
        case DType_F64:
            cuda_gather_kernel_double<<<blocks, threads>>>((double*)a, a_layout, b, b_layout, (double*)c, axis, len);
            break;
    }
}

#define DEFINE_CUDA_SCATTER_KERNEL(type) \
__global__ void cuda_scatter_kernel_##type( \
    type* a, Layout a_layout, uint32_t* b, Layout b_layout, type* c, Layout c_layout, size_t axis, int len \
) { \
    cuda_scatter_kernel<type>(a, a_layout, b, b_layout, c, c_layout, axis, len); \
}

DEFINE_CUDA_SCATTER_KERNEL(uint32_t)
DEFINE_CUDA_SCATTER_KERNEL(float)
DEFINE_CUDA_SCATTER_KERNEL(double)

extern "C" void cuda_scatter(
    DType dtype, void* a, Layout a_layout, uint32_t* b, Layout b_layout, void* c, Layout c_layout, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_scatter_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)a, a_layout, b, b_layout, (uint32_t*)c, c_layout, axis, len);
            break;
        case DType_F32:
            cuda_scatter_kernel_float<<<blocks, threads>>>((float*)a, a_layout, b, b_layout, (float*)c, c_layout, axis, len);
            break;
        case DType_F64:
            cuda_scatter_kernel_double<<<blocks, threads>>>((double*)a, a_layout, b, b_layout, (double*)c, c_layout, axis, len);
            break;
    }
}

#define DEFINE_CUDA_SCATTER_ADD_KERNEL(type) \
__global__ void cuda_scatter_add_kernel_##type( \
    type* a, Layout a_layout, uint32_t* b, Layout b_layout, type* c, Layout c_layout, size_t axis, int len \
) { \
    cuda_scatter_add_kernel<type>(a, a_layout, b, b_layout, c, c_layout, axis, len); \
}

DEFINE_CUDA_SCATTER_ADD_KERNEL(uint32_t)
DEFINE_CUDA_SCATTER_ADD_KERNEL(float)
DEFINE_CUDA_SCATTER_ADD_KERNEL(double)

extern "C" void cuda_scatter_add(
    DType dtype, void* a, Layout a_layout, uint32_t* b, Layout b_layout, void* c, Layout c_layout, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_scatter_add_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)a, a_layout, b, b_layout, (uint32_t*)c, c_layout, axis, len);
            break;
        case DType_F32:
            cuda_scatter_add_kernel_float<<<blocks, threads>>>((float*)a, a_layout, b, b_layout, (float*)c, c_layout, axis, len);
            break;
        case DType_F64:
            cuda_scatter_add_kernel_double<<<blocks, threads>>>((double*)a, a_layout, b, b_layout, (double*)c, c_layout, axis, len);
            break;
    }
}

#define DEFINE_CUDA_INDEX_SELECT_KERNEL(type) \
__global__ void cuda_index_select_kernel_##type( \
    type* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, type* output_data, Layout output_layout, size_t axis, int len \
) { \
    cuda_index_select_kernel<type>(input_data, input_layout, index_data, index_layout, output_data, output_layout, axis, len); \
}

DEFINE_CUDA_INDEX_SELECT_KERNEL(uint32_t)
DEFINE_CUDA_INDEX_SELECT_KERNEL(float)
DEFINE_CUDA_INDEX_SELECT_KERNEL(double)

extern "C" void cuda_index_select(
    DType dtype, void* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, void* src_data, Layout src_layout, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_index_select_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)input_data, input_layout, index_data, index_layout, (uint32_t*)src_data, src_layout, axis, len);
            break;
        case DType_F32:
            cuda_index_select_kernel_float<<<blocks, threads>>>((float*)input_data, input_layout, index_data, index_layout, (float*)src_data, src_layout, axis, len);
            break;
        case DType_F64:
            cuda_index_select_kernel_double<<<blocks, threads>>>((double*)input_data, input_layout, index_data, index_layout, (double*)src_data, src_layout, axis, len);
            break;
    }
}

#define DEFINE_CUDA_INDEX_COPY_KERNEL(type) \
__global__ void cuda_index_copy_kernel_##type( \
    type* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, type* src_data, Layout src_layout, NDimArray dest_shape, size_t axis, int len \
) { \
    cuda_index_copy_kernel<type>(input_data, input_layout, index_data, index_layout, src_data, src_layout, dest_shape, axis, len); \
}

DEFINE_CUDA_INDEX_COPY_KERNEL(uint32_t)
DEFINE_CUDA_INDEX_COPY_KERNEL(float)
DEFINE_CUDA_INDEX_COPY_KERNEL(double)

extern "C" void cuda_index_copy(
    DType dtype, void* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, void* src_data, Layout src_layout, NDimArray dest_shape, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_index_copy_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)input_data, input_layout, index_data, index_layout, (uint32_t*)src_data, src_layout, dest_shape, axis, len);
            break;
        case DType_F32:
            cuda_index_copy_kernel_float<<<blocks, threads>>>((float*)input_data, input_layout, index_data, index_layout, (float*)src_data, src_layout, dest_shape, axis, len);
            break;
        case DType_F64:
            cuda_index_copy_kernel_double<<<blocks, threads>>>((double*)input_data, input_layout, index_data, index_layout, (double*)src_data, src_layout, dest_shape, axis, len);
            break;
    }
}

#define DEFINE_CUDA_INDEX_ADD_KERNEL(type) \
__global__ void cuda_index_add_kernel_##type( \
    type* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, type* src_data, Layout src_layout, NDimArray dest_shape, size_t axis, int len \
) { \
    cuda_index_add_kernel<type>(input_data, input_layout, index_data, index_layout, src_data, src_layout, dest_shape, axis, len); \
}

DEFINE_CUDA_INDEX_ADD_KERNEL(uint32_t)
DEFINE_CUDA_INDEX_ADD_KERNEL(float)
DEFINE_CUDA_INDEX_ADD_KERNEL(double)

extern "C" void cuda_index_add(
    DType dtype, void* input_data, Layout input_layout, uint32_t* index_data, Layout index_layout, void* src_data, Layout src_layout, NDimArray dest_shape, size_t axis, int len
) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    switch (dtype) {
        case DType_U32:
            cuda_index_add_kernel_uint32_t<<<blocks, threads>>>((uint32_t*)input_data, input_layout, index_data, index_layout, (uint32_t*)src_data, src_layout, dest_shape, axis, len);
            break;
        case DType_F32:
            cuda_index_add_kernel_float<<<blocks, threads>>>((float*)input_data, input_layout, index_data, index_layout, (float*)src_data, src_layout, dest_shape, axis, len);
            break;
        case DType_F64:
            cuda_index_add_kernel_double<<<blocks, threads>>>((double*)input_data, input_layout, index_data, index_layout, (double*)src_data, src_layout, dest_shape, axis, len);
            break;
    }
}

#define DEFINE_CUDA_FILL_KERNEL(type) \
__global__ void cuda_fill_kernel_##type( \
    type* output_data, type value, int len \
) { \
    cuda_fill_kernel<type>(output_data, value, len); \
}

DEFINE_CUDA_FILL_KERNEL(uint32_t)
DEFINE_CUDA_FILL_KERNEL(float)
DEFINE_CUDA_FILL_KERNEL(double)

#define DEFINE_EXTERN_CUDA_FILL(type) \
extern "C" void cuda_fill_##type( \
    type* output_data, type value, int len \
) { \
    int threads = 256; \
    int blocks = (len + threads - 1) / threads; \
    cuda_fill_kernel_##type<<<blocks, threads>>>(output_data, value, len); \
}

DEFINE_EXTERN_CUDA_FILL(uint32_t)
DEFINE_EXTERN_CUDA_FILL(float)
DEFINE_EXTERN_CUDA_FILL(double)
