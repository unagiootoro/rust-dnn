use std::ffi::c_float;

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_im2col(
        img: *const c_float,
        img_len: i32,
        col: *mut c_float,
        col_len: i32,
        bsize: usize,
        ch: usize,
        img_h: usize,
        img_w: usize,
        out_h: usize,
        out_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_col2im(
        col: *const c_float,
        col_len: i32,
        img: *mut c_float,
        img_len: i32,
        bsize: usize,
        ch: usize,
        img_h: usize,
        img_w: usize,
        out_h: usize,
        out_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    );
}
