#include "utils.cuh"
#include "macro.cuh"
#include "common.cuh"

DEFINE_OP1_KERNEL2(cuda_neg_kernel, float, -)
DEFINE_OP1_KERNEL2(cuda_neg_kernel, double, -)

DEFINE_OP2_KERNEL2(cuda_add_kernel, uint32_t, +)
DEFINE_OP2_KERNEL2(cuda_add_kernel, float, +)
DEFINE_OP2_KERNEL2(cuda_add_kernel, double, +)

DEFINE_OP2_KERNEL2(cuda_sub_kernel, uint32_t, -)
DEFINE_OP2_KERNEL2(cuda_sub_kernel, float, -)
DEFINE_OP2_KERNEL2(cuda_sub_kernel, double, -)

DEFINE_OP2_KERNEL2(cuda_mul_kernel, uint32_t, *)
DEFINE_OP2_KERNEL2(cuda_mul_kernel, float, *)
DEFINE_OP2_KERNEL2(cuda_mul_kernel, double, *)

DEFINE_OP2_KERNEL2(cuda_div_kernel, uint32_t, /)
DEFINE_OP2_KERNEL2(cuda_div_kernel, float, /)
DEFINE_OP2_KERNEL2(cuda_div_kernel, double, /)

DEFINE_OP2_ASSIGN_KERNEL(copy_kernel, =)
DEFINE_OP2_ASSIGN_KERNEL(add_assign_kernel, +=)
DEFINE_CMP_OP2_KERNEL(lte_kernel, <=)
DEFINE_CMP_OP2_KERNEL(gt_kernel, >)
DEFINE_CMP_OP2_KERNEL(eq_kernel, ==)

__global__ void matmul_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    float* c,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_rows = a_shape.data[0];
        size_t a_cols = a_shape.data[1];
        size_t b_cols = b_shape.data[1];

        size_t i = idx / b_cols;
        size_t j = idx % b_cols;
        float sum = 0;
        for (size_t k = 0; k < a_cols; k++) {
            size_t a_idx = compute_offset_by_indices_dim2(a_base_offset, &a_strides.data[0], i, k);
            size_t b_idx = compute_offset_by_indices_dim2(b_base_offset, &b_strides.data[0], k, j);
            sum += a[a_idx] * b[b_idx];
        }
        c[i * b_cols + j] = sum;
    }
}

__global__ void cmp_max_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    float* c,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx);
        size_t b_idx = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx);
        if (a[a_idx] > b[b_idx]) {
            c[idx] = a[a_idx];
        } else {
            c[idx] = b[b_idx];
        }
    }
}


__global__ void cuda_contiguous_kernel_float(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx);
        b[idx] = a[a_idx];
    }
}

__global__ void cuda_contiguous_kernel_double(
    double* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    double* b,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t a_idx = compute_offset(a_base_offset, &a_shape.data[0], &a_strides.data[0], a_ndim, idx);
        b[idx] = a[a_idx];
    }
}

__global__ void set_item_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    NDimArray new_shape,
    size_t new_storage_offset,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t dst_idx = compute_offset(new_storage_offset, &new_shape.data[0], &a_strides.data[0], a_ndim, idx);
        size_t src_idx = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx);
        a[dst_idx] = b[src_idx];
    }
}

__global__ void get_item_by_index_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    float* c,
    size_t axis,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t b_index = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx);
        size_t b_value = b[b_index];
        size_t a_index = compute_offset_by_axis_index(a_base_offset, &b_shape.data[0], &a_strides.data[0], b_ndim, idx, axis, b_value);
        c[idx] = a[a_index];
    }
}

__global__ void set_item_by_index_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    float* c, size_t c_base_offset, NDimArray c_shape, NDimArray c_strides, size_t c_ndim,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = b_shape.data[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t b_index = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx);
            size_t b_value = b[b_index];
            size_t a_index = compute_offset_by_axis_index(a_base_offset, &b_shape.data[0], &a_strides.data[0], b_ndim, idx, axis, b_value);
            size_t c_index = compute_offset(c_base_offset, &c_shape.data[0], &c_strides.data[0], c_ndim, idx);
            a[a_index] = c[c_index];
        }
    }
}

__global__ void add_item_by_index_kernel(
    float* a, size_t a_base_offset, NDimArray a_shape, NDimArray a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, NDimArray b_shape, NDimArray b_strides, size_t b_ndim,
    float* c, size_t c_base_offset, NDimArray c_shape, NDimArray c_strides, size_t c_ndim,
    size_t axis,
    int len
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        int axis_dim = b_shape.data[axis];
        for (int j = 0; j < axis_dim; j++) {
            int idx = i + len * j;
            size_t b_index = compute_offset(b_base_offset, &b_shape.data[0], &b_strides.data[0], b_ndim, idx);
            size_t b_value = b[b_index];
            size_t a_index = compute_offset_by_axis_index(a_base_offset, &b_shape.data[0], &a_strides.data[0], b_ndim, idx, axis, b_value);
            size_t c_index = compute_offset(c_base_offset, &c_shape.data[0], &c_strides.data[0], c_ndim, idx);
            a[a_index] += c[c_index];
        }
    }
}

__global__ void fill_kernel(float* a, float value, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        a[idx] = value;
    }
}

extern "C" void copy(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    copy_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        len
    );
}

DEFINE_EXTERN_OP1_KERNEL(cuda_neg, float)
DEFINE_EXTERN_OP1_KERNEL(cuda_neg, double)

DEFINE_EXTERN_OP2_KERNEL(cuda_add, uint32_t)
DEFINE_EXTERN_OP2_KERNEL(cuda_add, float)
DEFINE_EXTERN_OP2_KERNEL(cuda_add, double)

extern "C" void add_assign(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    add_assign_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        len
    );
}

DEFINE_EXTERN_OP2_KERNEL(cuda_sub, uint32_t)
DEFINE_EXTERN_OP2_KERNEL(cuda_sub, float)
DEFINE_EXTERN_OP2_KERNEL(cuda_sub, double)

DEFINE_EXTERN_OP2_KERNEL(cuda_mul, uint32_t)
DEFINE_EXTERN_OP2_KERNEL(cuda_mul, float)
DEFINE_EXTERN_OP2_KERNEL(cuda_mul, double)

DEFINE_EXTERN_OP2_KERNEL(cuda_div, uint32_t)
DEFINE_EXTERN_OP2_KERNEL(cuda_div, float)
DEFINE_EXTERN_OP2_KERNEL(cuda_div, double)

extern "C" void matmul(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    matmul_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        len
    );
}

extern "C" void cuda_lte(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    lte_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        len
    );
}

extern "C" void cuda_gt(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    gt_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        len
    );
}

extern "C" void cuda_eq(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    eq_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        len
    );
}

extern "C" void cuda_cmp_max(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    cmp_max_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        len
    );
}

extern "C" void cuda_contiguous_float(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    cuda_contiguous_kernel_float<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b,
        len
    );
}

extern "C" void cuda_contiguous_double(
    double* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    double* b,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    cuda_contiguous_kernel_double<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b,
        len
    );
}

extern "C" void set_item(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    size_t* new_shape,
    size_t new_storage_offset,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);
    NDimArray new_shape_array(new_shape, a_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    set_item_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        new_shape_array,
        new_storage_offset,
        len
    );
}

extern "C" void get_item_by_index(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c,
    size_t axis,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    get_item_by_index_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c,
        axis,
        len
    );
}

extern "C" void set_item_by_index(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c, size_t c_base_offset, size_t* c_shape, size_t* c_strides, size_t c_ndim,
    size_t axis,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);
    NDimArray c_shape_array(c_shape, c_ndim);
    NDimArray c_strides_array(c_strides, c_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    set_item_by_index_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c, b_base_offset, c_shape_array, c_strides_array, c_ndim,
        axis,
        len
    );
}

extern "C" void add_item_by_index(
    float* a, size_t a_base_offset, size_t* a_shape, size_t* a_strides, size_t a_ndim,
    float* b, size_t b_base_offset, size_t* b_shape, size_t* b_strides, size_t b_ndim,
    float* c, size_t c_base_offset, size_t* c_shape, size_t* c_strides, size_t c_ndim,
    size_t axis,
    int len
) {
    NDimArray a_shape_array(a_shape, a_ndim);
    NDimArray a_strides_array(a_strides, a_ndim);
    NDimArray b_shape_array(b_shape, b_ndim);
    NDimArray b_strides_array(b_strides, b_ndim);
    NDimArray c_shape_array(c_shape, c_ndim);
    NDimArray c_strides_array(c_strides, c_ndim);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    add_item_by_index_kernel<<<blocks, threads>>>(
        a, a_base_offset, a_shape_array, a_strides_array, a_ndim,
        b, b_base_offset, b_shape_array, b_strides_array, b_ndim,
        c, b_base_offset, c_shape_array, c_strides_array, c_ndim,
        axis,
        len
    );
}

extern "C" void cuda_fill(float* a, float value, int len) {
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    fill_kernel<<<blocks, threads>>>(a, value, len);
}
