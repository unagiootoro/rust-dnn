#include "utils.cuh"

__device__ size_t compute_offset(size_t base_offset, size_t* shape, size_t* strides, size_t ndim, size_t linear_index) {
    size_t offset = 0;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] > 0) {
            size_t idx = linear_index % shape[i];
            offset += idx * strides[i];
        }
        linear_index /= shape[i];
    }
    return base_offset + offset;
}

__device__ size_t compute_offset2(Layout* layout, size_t linear_index) {
    size_t offset = 0;
    for (int i = layout->ndim - 1; i >= 0; i--) {
        if (layout->stride[i] > 0) {
            size_t idx = linear_index % layout->shape[i];
            offset += idx * layout->stride[i];
        }
        linear_index /= layout->shape[i];
    }
    return layout->storage_offset + offset;
}

__device__ size_t compute_offset_by_indices_dim2(size_t base_offset, size_t* strides, size_t i, size_t j) {
    size_t offset = 0;
    if (strides[0] > 0) {
        offset += i * strides[0];
    }
    if (strides[1] > 0) {
        offset += j * strides[1];
    }
    return base_offset + offset;
}

__device__ size_t compute_offset_by_ranges(size_t base_offset, size_t* strides, size_t* ranges, size_t ndim) {
    size_t offset = 0;
    for (size_t i = 0; i < ndim; i++) {
        if (strides[i] > 0) {
            offset += ranges[i * 2] * strides[i];
        }
    }
    return base_offset + offset;
}

__device__ void unravel_index(size_t* shape, size_t ndim, size_t linear_index, size_t* indices) {
    for (int i = ndim - 1; i >= 0; i--) {
        size_t idx = linear_index % shape[i];
        linear_index /= shape[i];
        indices[i] = idx;
    }
}

__device__ size_t compute_offset_by_indices(size_t base_offset, size_t* strides, size_t ndim, size_t* indices) {
    size_t offset = 0;
    for (size_t i = 0; i < ndim; i++) {
        if (strides[i] > 0) {
            offset += indices[i] * strides[i];
        }
    }
    return base_offset + offset;
}

__device__ size_t compute_offset_by_axis_index(size_t base_offset, size_t* shape, size_t* strides, size_t ndim, size_t linear_index, size_t axis, size_t index) {
    size_t offset = 0;
    for (int i = ndim - 1; i >= 0; i--) {
        if (strides[i] > 0) {
            size_t idx;
            if (i == axis) {
                idx = index;
            } else {
                idx = linear_index % shape[i];
            }
            offset += idx * strides[i];
        }
        linear_index /= shape[i];
    }
    return base_offset + offset;
}
