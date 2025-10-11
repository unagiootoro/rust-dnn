#pragma once

#define MAX_NDIM (8)

class NDimArray {
public:
    size_t data[MAX_NDIM];

    NDimArray(size_t* data, size_t len) {
        memcpy(&this->data[0], data, len * sizeof(size_t));
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
