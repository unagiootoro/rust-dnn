__global__ void im2col_kernel(
    float* img,
    int img_len,
    float* col,
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

__global__ void col2im_kernel(
    float* col,
    int col_len,
    float* img,
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

extern "C" void cuda_im2col(
    float* img,
    int img_len,
    float* col,
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
    im2col_kernel<<<blocks, threads>>>(
        img,
        img_len,
        col,
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

extern "C" void cuda_col2im(
    float* col,
    int col_len,
    float* img,
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
    col2im_kernel<<<blocks, threads>>>(
        col,
        col_len,
        img,
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
