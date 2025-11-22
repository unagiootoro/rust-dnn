template <typename T>
__device__ void cuda_im2col_kernel(
    T* img,
    int img_len,
    T* col,
    int col_len,
    size_t bsize,
    size_t ch,
    size_t img_h,
    size_t img_w,
    size_t out_h,
    size_t out_w,
    size_t fil_h,
    size_t fil_w,
    size_t stride_h,
    size_t stride_w
) {
    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_idx < col_len) {
        size_t n = col_idx / (ch * fil_h * fil_w * out_h * out_w);
        size_t m = col_idx / (fil_h * fil_w * out_h * out_w) % ch;
        size_t k = col_idx / (fil_w * out_h * out_w) % fil_h;
        size_t l = col_idx / (out_h * out_w) % fil_w;
        size_t i2 = col_idx / out_w % out_h;
        size_t j2 = col_idx % out_w;

        size_t i = i2 * stride_h;
        size_t j = j2 * stride_w;

        size_t img_idx = n * (ch * img_h * img_w);
        img_idx += m * (img_h * img_w);
        img_idx += (i + k) * img_w;
        img_idx += j + l;

        if (img_idx < img_len) {
            col[col_idx] = img[img_idx];
        }
    }
}

template <typename T>
__device__ void cuda_col2im_kernel(
    T* col,
    int col_len,
    T* img,
    int img_len,
    size_t bsize,
    size_t ch,
    size_t img_h,
    size_t img_w,
    size_t out_h,
    size_t out_w,
    size_t fil_h,
    size_t fil_w,
    size_t stride_h,
    size_t stride_w,
    int len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        size_t n = idx / (ch * out_h * out_w);
        size_t m = idx / (out_h * out_w) % ch;
        size_t i2 = idx / out_w % out_h;
        size_t j2 = idx % out_w;

        size_t i = i2 * stride_h;
        size_t j = j2 * stride_w;

        for (int k = 0; k < fil_h; k++) {
            for (int l = 0; l < fil_w; l++) {
                size_t img_idx = n * (ch * img_h * img_w);
                img_idx += m * (img_h * img_w);
                img_idx += (i + k) * img_w;
                img_idx += j + l;

                size_t col_idx = n * (ch * fil_h * fil_w * out_h * out_w);
                col_idx += m * (fil_h * fil_w * out_h * out_w);
                col_idx += k * (fil_w * out_h * out_w);
                col_idx += l * (out_h * out_w);
                col_idx += (i / stride_h) * out_w;
                col_idx += (j / stride_w);

                if (col_idx < col_len && img_idx < img_len) {
                    img[img_idx] += col[col_idx];
                }
            }
        }
    }
}

#define DEFINE_CUDA_IM2COL_KERNEL(type) \
__global__ void cuda_im2col_kernel_##type( \
    type* img, \
    int img_len, \
    type* col, \
    int col_len, \
    size_t bsize, \
    size_t ch, \
    size_t img_h, \
    size_t img_w, \
    size_t out_h, \
    size_t out_w, \
    size_t fil_h, \
    size_t fil_w, \
    size_t stride_h, \
    size_t stride_w \
) { \
    cuda_im2col_kernel<type>( \
        img, \
        img_len, \
        col, \
        col_len, \
        bsize,  \
        ch,  \
        img_h,  \
        img_w,  \
        out_h,  \
        out_w,  \
        fil_h,  \
        fil_w,  \
        stride_h,  \
        stride_w \
    ); \
} \

DEFINE_CUDA_IM2COL_KERNEL(float)
DEFINE_CUDA_IM2COL_KERNEL(double)

extern "C" void cuda_im2col(
    int dtype,
    void* img,
    int img_len,
    void* col,
    int col_len,
    size_t bsize,
    size_t ch,
    size_t img_h,
    size_t img_w,
    size_t out_h,
    size_t out_w,
    size_t fil_h,
    size_t fil_w,
    size_t stride_h,
    size_t stride_w
) {
    int threads = 256;
    int blocks = (col_len + threads - 1) / threads;
    if (dtype == 1) {
        cuda_im2col_kernel_float<<<blocks, threads>>>(
            (float*)img,
            img_len,
            (float*)col,
            col_len,
            bsize, 
            ch, 
            img_h, 
            img_w, 
            out_h, 
            out_w, 
            fil_h, 
            fil_w, 
            stride_h, 
            stride_w
        );
    } else if (dtype == 2) {
        cuda_im2col_kernel_double<<<blocks, threads>>>(
            (double*)img,
            img_len,
            (double*)col,
            col_len,
            bsize, 
            ch, 
            img_h, 
            img_w, 
            out_h, 
            out_w, 
            fil_h, 
            fil_w, 
            stride_h, 
            stride_w
        );
    }
}

#define DEFINE_CUDA_COL2IM_KERNEL(type) \
__global__ void cuda_col2im_kernel_##type( \
    type* col, \
    int col_len, \
    type* img, \
    int img_len, \
    size_t bsize, \
    size_t ch, \
    size_t img_h, \
    size_t img_w, \
    size_t out_h, \
    size_t out_w, \
    size_t fil_h, \
    size_t fil_w, \
    size_t stride_h, \
    size_t stride_w, \
    int len \
) { \
    cuda_col2im_kernel<type>( \
        col, \
        col_len, \
        img, \
        img_len,  \
        bsize,  \
        ch,  \
        img_h,  \
        img_w,  \
        out_h,  \
        out_w,  \
        fil_h,  \
        fil_w,  \
        stride_h,  \
        stride_w,  \
        len \
    ); \
}

DEFINE_CUDA_COL2IM_KERNEL(float)
DEFINE_CUDA_COL2IM_KERNEL(double)

extern "C" void cuda_col2im(
    int dtype,
    void* col,
    int col_len,
    void* img,
    int img_len,
    size_t bsize,
    size_t ch,
    size_t img_h,
    size_t img_w,
    size_t out_h,
    size_t out_w,
    size_t fil_h,
    size_t fil_w,
    size_t stride_h,
    size_t stride_w
) {
    int threads = 256;
    int blocks = (col_len + threads - 1) / threads;
    int len = bsize * ch * out_h * out_w;
    if (dtype == 1) {
        cuda_col2im_kernel_float<<<blocks, threads>>>(
            (float*)col,
            col_len,
            (float*)img,
            img_len, 
            bsize, 
            ch, 
            img_h, 
            img_w, 
            out_h, 
            out_w, 
            fil_h, 
            fil_w, 
            stride_h, 
            stride_w, 
            len
        );
    } else if (dtype == 2) {
        cuda_col2im_kernel_double<<<blocks, threads>>>(
            (double*)col,
            col_len,
            (double*)img,
            img_len, 
            bsize, 
            ch, 
            img_h, 
            img_w, 
            out_h, 
            out_w, 
            fil_h, 
            fil_w, 
            stride_h, 
            stride_w, 
            len
        );
    }
}
