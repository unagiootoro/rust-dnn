use libc::{c_double, size_t};
use std::ffi::c_float;

#[macro_export]
macro_rules! define_extern_op1_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
        #[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const c_float, a_base_offset: size_t,
                b: *mut c_float,
                len: i32
            );
        }

        #[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const c_double, a_base_offset: size_t,
                b: *mut c_double,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_op2_func {
    ($u32_fn_name: ident, $f32_fn_name: ident, $f64_fn_name: ident) => {
        #[rustfmt::skip]
        unsafe extern "C" {
            pub fn $u32_fn_name(
                a: *const u32, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
                b: *const u32, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
                c: *mut u32,
                len: i32
            );
        }

        #[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
                b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
                c: *mut c_float,
                len: i32
            );
        }

        #[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const c_double, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
                b: *const c_double, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
                c: *mut c_double,
                len: i32
            );
        }
    };
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn copy(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        len: i32
    );
}

define_extern_op1_func!(cuda_neg_float, cuda_neg_double);
define_extern_op2_func!(cuda_add_uint32_t, cuda_add_float, cuda_add_double);

#[rustfmt::skip]
unsafe extern "C" {
    pub fn add_assign(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *const c_float, b_base_offset: size_t, b_shape: *const size_t, b_strides: *const size_t, b_ndim: size_t,
        len: i32
    );
}

define_extern_op2_func!(cuda_sub_uint32_t, cuda_sub_float, cuda_sub_double);
define_extern_op2_func!(cuda_mul_uint32_t, cuda_mul_float, cuda_mul_double);
define_extern_op2_func!(cuda_div_uint32_t, cuda_div_float, cuda_div_double);

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
    pub fn cuda_contiguous_float(
        a: *const c_float, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_float,
        len: i32
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_contiguous_double(
        a: *const c_double, a_base_offset: size_t, a_shape: *const size_t, a_strides: *const size_t, a_ndim: size_t,
        b: *mut c_double,
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
