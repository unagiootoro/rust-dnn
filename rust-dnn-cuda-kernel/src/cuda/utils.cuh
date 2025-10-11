#pragma once

#include "common.cuh"

__device__ size_t compute_offset(size_t base_offset, size_t* shape, size_t* strides, size_t ndim, size_t linear_index);
__device__ size_t compute_offset2(Layout* layout, size_t linear_index);
__device__ size_t compute_offset_by_indices_dim2(size_t base_offset, size_t* strides, size_t i, size_t j);
__device__ size_t compute_offset_by_ranges(size_t base_offset, size_t* strides, size_t* ranges, size_t ndim);
__device__ void unravel_index(size_t base_offset, size_t* shape, size_t ndim, size_t linear_index, size_t* indices);
__device__ size_t compute_offset_by_indices(size_t base_offset, size_t* strides, size_t ndim, size_t* indices);
__device__ size_t compute_offset_by_axis_index(size_t base_offset, size_t* shape, size_t* strides, size_t ndim, size_t linear_index, size_t axis, size_t index);
