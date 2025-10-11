use std::ffi::c_float;

use rust_dnn_cuda_kernel::basic::{
    cuda_add_double, cuda_add_float, cuda_add_uint32_t, cuda_contiguous_double,
    cuda_contiguous_float, cuda_div_double, cuda_div_float, cuda_div_uint32_t, cuda_mul_double,
    cuda_mul_float, cuda_mul_uint32_t, cuda_neg_double, cuda_neg_float, cuda_sub_double,
    cuda_sub_float, cuda_sub_uint32_t,
};
use rust_dnn_cuda_kernel::clayout::{CLayout, MAX_NDIM};
use rust_dnn_cuda_kernel::cuda::check_cuda_error;
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;
use rust_dnn_cuda_kernel::math::{
    cuda_ln_double, cuda_ln_float, cuda_pow_double, cuda_pow_float, cuda_sum_axis_double, cuda_sum_axis_float
};

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};
use libc::size_t;

pub(crate) type CudaOp1Func<T> = unsafe fn(a: *const T, a_base_offset: size_t, b: *mut T, len: i32);

pub(crate) type CudaOp2Func<T> =
    unsafe fn(a: *const T, a_layout: CLayout, b: *const T, b_layout: CLayout, c: *mut T, len: i32);

pub(crate) type CudaOp2AssignFunc = unsafe extern "C" fn(
    a: *const c_float,
    a_base_offset: size_t,
    a_shape: *const size_t,
    a_strides: *const size_t,
    a_ndim: size_t,
    b: *const c_float,
    b_base_offset: size_t,
    b_shape: *const size_t,
    b_strides: *const size_t,
    b_ndim: size_t,
    len: i32,
);

#[derive(Debug, Clone, Copy)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn contiguous<T: Num>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        let input_data = storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(layout.len());
            match T::dtype() {
                DType::F32 => cuda_contiguous_float(
                    input_data.ptr() as *const f32,
                    layout.storage_offset(),
                    layout.shape().as_ptr() as *const size_t,
                    layout.stride().as_ptr() as *const size_t,
                    layout.ndim(),
                    output_data.ptr() as *mut f32,
                    layout.len() as i32,
                ),
                DType::F64 => cuda_contiguous_double(
                    input_data.ptr() as *const f64,
                    layout.storage_offset(),
                    layout.shape().as_ptr() as *const size_t,
                    layout.stride().as_ptr() as *const size_t,
                    layout.ndim(),
                    output_data.ptr() as *mut f64,
                    layout.len() as i32,
                ),
                _ => panic!(),
            }
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn sum_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let input_data = input_storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(output_layout.len());
            let input_clayout = layout_to_clayout(input_layout)?;
            let output_clayout = layout_to_clayout(output_layout)?;
            match T::dtype() {
                DType::F32 => {
                    cuda_sum_axis_float(
                        input_data.ptr() as *const f32,
                        input_clayout,
                        output_data.ptr() as *mut f32,
                        output_clayout,
                        axis,
                        output_layout.len() as i32,
                    );
                }
                DType::F64 => {
                    cuda_sum_axis_double(
                        input_data.ptr() as *const f64,
                        input_clayout,
                        output_data.ptr() as *mut f64,
                        output_clayout,
                        axis,
                        output_layout.len() as i32,
                    );
                }
                _ => panic!(),
            }
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn op_add<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_add)
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_sub)
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_mul)
    }

    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_div)
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_neg)
    }

    fn op_pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_pow)
    }

    fn op_ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_ln)
    }
}

#[macro_export]
macro_rules! define_cuda_op1_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(a: *const T, a_base_offset: size_t, b: *mut T, len: i32) {
            unsafe {
                match T::dtype() {
                    DType::F32 => $fn_name_f32(a as *const f32, a_base_offset, b as *mut f32, len),
                    DType::F64 => $fn_name_f64(a as *const f64, a_base_offset, b as *mut f64, len),
                    _ => panic!(),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_cuda_op2_func {
    ($fn_name: ident, $fn_name_u32: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(
            a: *const T,
            a_layout: CLayout,
            b: *const T,
            b_layout: CLayout,
            c: *mut T,
            len: i32,
        ) {
            unsafe {
                match T::dtype() {
                    DType::U32 => $fn_name_u32(
                        a as *const u32,
                        a_layout,
                        b as *const u32,
                        b_layout,
                        c as *mut u32,
                        len,
                    ),
                    DType::F32 => $fn_name_f32(
                        a as *const f32,
                        a_layout,
                        b as *const f32,
                        b_layout,
                        c as *mut f32,
                        len,
                    ),
                    DType::F64 => $fn_name_f64(
                        a as *const f64,
                        a_layout,
                        b as *const f64,
                        b_layout,
                        c as *mut f64,
                        len,
                    ),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_cuda_float_op2_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Float>(
            a: *const T,
            a_layout: CLayout,
            b: *const T,
            b_layout: CLayout,
            c: *mut T,
            len: i32,
        ) {
            unsafe {
                match T::dtype() {
                    DType::U32 => panic!(),
                    DType::F32 => $fn_name_f32(
                        a as *const f32,
                        a_layout,
                        b as *const f32,
                        b_layout,
                        c as *mut f32,
                        len,
                    ),
                    DType::F64 => $fn_name_f64(
                        a as *const f64,
                        a_layout,
                        b as *const f64,
                        b_layout,
                        c as *mut f64,
                        len,
                    ),
                }
            }
        }
    };
}

define_cuda_op1_func!(cuda_neg, cuda_neg_float, cuda_neg_double);
define_cuda_op2_func!(cuda_add, cuda_add_uint32_t, cuda_add_float, cuda_add_double);
define_cuda_op2_func!(cuda_sub, cuda_sub_uint32_t, cuda_sub_float, cuda_sub_double);
define_cuda_op2_func!(cuda_mul, cuda_mul_uint32_t, cuda_mul_float, cuda_mul_double);
define_cuda_op2_func!(cuda_div, cuda_div_uint32_t, cuda_div_float, cuda_div_double);
define_cuda_float_op2_func!(cuda_pow, cuda_pow_float, cuda_pow_double);
define_cuda_op1_func!(cuda_ln, cuda_ln_float, cuda_ln_double);

pub(crate) fn cuda_op1_func_call<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f: CudaOp1Func<T>,
) -> Result<Storage<T>> {
    let input_data = storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(layout.len());
        f(
            input_data.ptr() as *const T,
            layout.storage_offset(),
            output_data.ptr() as *mut T,
            layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

pub(crate) fn cuda_op2_func_call<T: Num>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaOp2Func<T>,
) -> Result<Storage<T>> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let lhs_clayout = layout_to_clayout(lhs_layout)?;
    let rhs_clayout = layout_to_clayout(rhs_layout)?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(lhs_layout.len());
        f(
            lhs_data.ptr() as *const T,
            lhs_clayout,
            rhs_data.ptr() as *mut T,
            rhs_clayout,
            output_data.ptr() as *mut T,
            lhs_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

fn layout_to_clayout(layout: &Layout) -> Result<CLayout> {
    if layout.ndim() > MAX_NDIM {
        return Err(Error::ArgumentsError {
            msg: format!("Invalid layout ndim({})", layout.ndim()).to_string(),
        });
    }
    let mut shape = [0; MAX_NDIM];
    for (i, dim) in layout.shape().iter().enumerate() {
        shape[i] = *dim;
    }
    let mut stride = [0; MAX_NDIM];
    for (i, dim) in layout.stride().iter().enumerate() {
        stride[i] = *dim;
    }
    let clayout = CLayout {
        shape,
        stride,
        ndim: layout.ndim(),
        len: layout.len(),
        storage_offset: layout.storage_offset(),
    };
    Ok(clayout)
}
