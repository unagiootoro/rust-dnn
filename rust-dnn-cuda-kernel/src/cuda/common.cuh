#pragma once

#define MAX_NDIM (8)

class NDimArray {
public:
    size_t data[MAX_NDIM];

    NDimArray(size_t* data, size_t len) {
        memcpy(&this->data[0], data, len * sizeof(size_t));
    }
};
