use libc::size_t;
use std::ffi::c_float;

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sum(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sum_axis(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_max_axis(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_float, b_shape: *const size_t,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_argmax_axis(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_float, b_shape: *const size_t,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_exp(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_log(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sqrt(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_pow(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        rhs: f32,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sin(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_cos(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_tanh(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}
