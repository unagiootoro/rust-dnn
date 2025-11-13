use libc::{c_double, size_t};
use std::ffi::c_float;

use crate::{
    clayout::CLayout, define_extern_float_op2_func, define_extern_op1_func,
    define_extern_reduce_axis_func, define_extern_reduce_cmp_axis_index_func,
    define_extern_reduce_func,
};

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

define_extern_op1_func!(cuda_exp_float, cuda_exp_double);
define_extern_op1_func!(cuda_ln_float, cuda_ln_double);
define_extern_op1_func!(cuda_sqrt_float, cuda_sqrt_double);
define_extern_op1_func!(cuda_sin_float, cuda_sin_double);
define_extern_op1_func!(cuda_cos_float, cuda_cos_double);
define_extern_op1_func!(cuda_tanh_float, cuda_tanh_double);

define_extern_float_op2_func!(cuda_pow_float, cuda_pow_double);

define_extern_reduce_func!(cuda_sum_float, cuda_sum_double);
define_extern_reduce_func!(cuda_max_float, cuda_max_double);

define_extern_reduce_axis_func!(cuda_sum_axis_float, cuda_sum_axis_double);
define_extern_reduce_axis_func!(cuda_max_axis_float, cuda_max_axis_double);

define_extern_reduce_cmp_axis_index_func!(cuda_argmax_axis_float, cuda_argmax_axis_double);
