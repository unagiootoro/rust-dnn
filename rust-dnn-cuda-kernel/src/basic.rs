use libc::{c_double, size_t, uint32_t};
use std::ffi::{c_float, c_void};

use crate::clayout::{CLayout, NDimArray};

#[macro_export]
macro_rules! define_extern_op1_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const c_float, a_storage_offset: usize,
                b: *mut c_float,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const c_double, a_storage_offset: usize,
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
                a: *const u32, a_layout: CLayout,
                b: *const u32, b_layout: CLayout,
                c: *mut u32,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *const f32, b_layout: CLayout,
                c: *mut c_float,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *const f64, b_layout: CLayout,
                c: *mut c_double,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_op2_u32_func {
    ($u32_fn_name: ident, $f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $u32_fn_name(
                a: *const u32, a_layout: CLayout,
                b: *const u32, b_layout: CLayout,
                c: *mut u32,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *const f32, b_layout: CLayout,
                c: *mut u32,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *const f64, b_layout: CLayout,
                c: *mut u32,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_op2_assign_func {
    ($u32_fn_name: ident, $f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $u32_fn_name(
                a: *mut u32, a_layout: CLayout,
                b: *const u32, b_layout: CLayout,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *mut f32, a_layout: CLayout,
                b: *const f32, b_layout: CLayout,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *mut f64, a_layout: CLayout,
                b: *const f64, b_layout: CLayout,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_float_op2_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *const f32, b_layout: CLayout,
                c: *mut c_float,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *const f64, b_layout: CLayout,
                c: *mut c_double,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_reduce_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *const f32,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *const f64,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_reduce_axis_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *mut f32, b_layout: CLayout,
                axis: usize,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *mut f64, b_layout: CLayout,
                axis: usize,
                len: i32
            );
        }
    };
}

#[macro_export]
macro_rules! define_extern_reduce_cmp_axis_index_func {
    ($f32_fn_name: ident, $f64_fn_name: ident) => {
#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f32_fn_name(
                a: *const f32, a_layout: CLayout,
                b: *mut u32, b_layout: CLayout,
                axis: usize,
                len: i32
            );
        }

#[rustfmt::skip]
        unsafe extern "C" {
            pub fn $f64_fn_name(
                a: *const f64, a_layout: CLayout,
                b: *mut u32, b_layout: CLayout,
                axis: usize,
                len: i32
            );
        }
    };
}

define_extern_op1_func!(cuda_neg_float, cuda_neg_double);
define_extern_op2_func!(cuda_add_uint32_t, cuda_add_float, cuda_add_double);
define_extern_op2_func!(cuda_sub_uint32_t, cuda_sub_float, cuda_sub_double);
define_extern_op2_func!(cuda_mul_uint32_t, cuda_mul_float, cuda_mul_double);
define_extern_op2_func!(cuda_div_uint32_t, cuda_div_float, cuda_div_double);

define_extern_op2_assign_func!(cuda_copy_uint32_t, cuda_copy_float, cuda_copy_double);
define_extern_op2_assign_func!(
    cuda_add_assign_uint32_t,
    cuda_add_assign_float,
    cuda_add_assign_double
);
define_extern_op2_assign_func!(
    cuda_sub_assign_uint32_t,
    cuda_sub_assign_float,
    cuda_sub_assign_double
);
define_extern_op2_assign_func!(
    cuda_mul_assign_uint32_t,
    cuda_mul_assign_float,
    cuda_mul_assign_double
);
define_extern_op2_assign_func!(
    cuda_div_assign_uint32_t,
    cuda_div_assign_float,
    cuda_div_assign_double
);

define_extern_op2_u32_func!(cuda_lt_uint32_t, cuda_lt_float, cuda_lt_double);
define_extern_op2_u32_func!(cuda_le_uint32_t, cuda_le_float, cuda_le_double);
define_extern_op2_u32_func!(cuda_gt_uint32_t, cuda_gt_float, cuda_gt_double);
define_extern_op2_u32_func!(cuda_ge_uint32_t, cuda_ge_float, cuda_ge_double);
define_extern_op2_u32_func!(cuda_eq_uint32_t, cuda_eq_float, cuda_eq_double);

unsafe extern "C" {
    pub fn cuda_convert(
        dtype1: i32,
        dtype2: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *mut c_void,
        len: i32,
    );
}

#[rustfmt::skip]
unsafe extern "C" {
    pub fn cuda_matmul(
        dtype: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *const c_void,
        b_layout: CLayout,
        c: *mut c_void,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_contiguous(
        dtype: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *mut c_void,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_gather(
        dtype: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *const u32,
        b_layout: CLayout,
        c: *mut c_void,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_scatter(
        dtype: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *const u32,
        b_layout: CLayout,
        c: *mut c_void,
        c_layout: CLayout,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_scatter_add(
        dtype: i32,
        a: *const c_void,
        a_layout: CLayout,
        b: *const u32,
        b_layout: CLayout,
        c: *mut c_void,
        c_layout: CLayout,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_index_select(
        dtype: i32,
        input_data: *const c_void,
        input_layout: CLayout,
        index_data: *const u32,
        index_layout: CLayout,
        output_data: *mut c_void,
        output_layout: CLayout,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_index_copy(
        dtype: i32,
        input_data: *const c_void,
        input_layout: CLayout,
        index_data: *const u32,
        index_layout: CLayout,
        src_data: *mut c_void,
        src_layout: CLayout,
        dest_shape: NDimArray,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_index_add(
        dtype: i32,
        input_data: *const c_void,
        input_layout: CLayout,
        index_data: *const u32,
        index_layout: CLayout,
        src_data: *mut c_void,
        src_layout: CLayout,
        dest_shape: NDimArray,
        axis: usize,
        len: i32,
    );
}

unsafe extern "C" {
    pub fn cuda_fill_uint32_t(output_data: *mut u32, value: u32, len: i32);
}

unsafe extern "C" {
    pub fn cuda_fill_float(output_data: *mut f32, value: f32, len: i32);
}

unsafe extern "C" {
    pub fn cuda_fill_double(output_data: *mut f64, value: f64, len: i32);
}
