use libc::{c_double, size_t};
use std::ffi::c_float;

use crate::{clayout::CLayout, define_extern_float_op2_func, define_extern_op1_func};

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
    pub fn cuda_sum_axis_float(
        a: *const c_float, a_layout: CLayout,
        b: *mut c_float, b_layout: CLayout,
        axis: size_t,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sum_axis_double(
        a: *const c_double, a_layout: CLayout,
        b: *mut c_double, b_layout: CLayout,
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

define_extern_op1_func!(cuda_ln_float, cuda_ln_double);

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_sqrt(
        a: *const c_float, a_base_offset: size_t,
        b: *mut c_float,
        len: i32
    );
}

define_extern_float_op2_func!(cuda_pow_float, cuda_pow_double);

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
