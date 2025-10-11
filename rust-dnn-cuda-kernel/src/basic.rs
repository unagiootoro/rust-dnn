use libc::size_t;
use std::ffi::c_float;

#[rustfmt::skip]
unsafe extern "C" {
    pub fn copy(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn neg(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn add(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn add_assign(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn sub(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn mul(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_div(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn matmul(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_lte(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_gt(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_eq(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_cmp_max(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn contiguous(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn set_item(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        new_shape: *const size_t,
        new_storage_offset: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn get_item_by_index(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *mut c_float,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn set_item_by_index(
        a: *mut c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *const c_float, c_base_offset: size_t, c_shape: *const size_t, c_strides: *const size_t, c_ndim: size_t,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn add_item_by_index(
        a: *mut c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        c: *const c_float, c_base_offset: size_t, c_shape: *const size_t, c_strides: *const size_t, c_ndim: size_t,
        axis: size_t,
        len: i32
    );
}

unsafe extern "C" {
    pub fn cuda_fill(a: *mut c_float, value: f32, len: i32);
}
