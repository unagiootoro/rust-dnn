#pragma once

#define MAX_NDIM (8)

enum DType {
    DType_U32 = 0,
    DType_F32 = 1,
    DType_F64 = 2
};

class NDimArray {
public:
    size_t data[MAX_NDIM];
    size_t ndim;

    NDimArray(size_t* data, size_t ndim) {
        memcpy(&this->data[0], data, ndim * sizeof(size_t));
        this->ndim = ndim;
    }
};

class Layout {
public:
    size_t shape[MAX_NDIM];
    size_t stride[MAX_NDIM];
    size_t ndim;
    size_t len;
    size_t storage_offset;

    Layout(size_t* shape, size_t* stride, size_t ndim, size_t len, size_t storage_offset) {
        memcpy(&this->shape[0], shape, ndim * sizeof(size_t));
        memcpy(&this->stride[0], stride, ndim * sizeof(size_t));
        this->ndim = ndim;
        this->len = len;
        this->storage_offset = storage_offset;
    }
};
