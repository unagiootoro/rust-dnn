use std::ffi::c_float;

use rust_dnn_cuda_kernel::basic::{add, contiguous, cuda_div, mul, neg, sub};
use rust_dnn_cuda_kernel::cuda::check_cuda_error;
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;
use rust_dnn_cuda_kernel::math::{cuda_pow, cuda_sum_axis};

use crate::backend::Backend;
use crate::error::Result;
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};
use libc::size_t;

pub(crate) type CudaOp1Func =
    unsafe extern "C" fn(a: *const c_float, a_base_offset: size_t, b: *mut c_float, len: i32);

pub(crate) type CudaOp2Func = unsafe extern "C" fn(
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
    c: *mut c_float,
    len: i32,
);

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
            contiguous(
                input_data.ptr() as *const f32,
                layout.storage_offset(),
                layout.shape().as_ptr() as *const size_t,
                layout.stride().as_ptr() as *const size_t,
                layout.ndim(),
                output_data.ptr() as *mut f32,
                layout.len() as i32,
            );
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
            cuda_sum_axis(
                input_data.ptr() as *const f32,
                input_layout.storage_offset(),
                input_layout.shape().as_ptr() as *const size_t,
                input_layout.stride().as_ptr() as *const size_t,
                input_layout.ndim(),
                output_data.ptr() as *mut f32,
                output_layout.storage_offset(),
                output_layout.shape().as_ptr() as *const size_t,
                output_layout.stride().as_ptr() as *const size_t,
                output_layout.ndim(),
                axis,
                output_layout.len() as i32,
            );
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
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, add)
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, sub)
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, mul)
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
        cuda_op1_func_call(storage, layout, neg)
    }

    fn op_pow_scalar<T: Float>(
        storage: &Storage<T>,
        layout: &Layout,
        rhs: T,
    ) -> Result<Storage<T>> {
        let input_data = storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(layout.len());
            cuda_pow(
                input_data.ptr() as *const f32,
                layout.storage_offset(),
                output_data.ptr() as *mut f32,
                rhs.as_f32(),
                layout.len() as i32,
            );
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }
}

pub(crate) fn cuda_op1_func_call<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f: CudaOp1Func,
) -> Result<Storage<T>> {
    let input_data = storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(layout.len());
        f(
            input_data.ptr() as *const f32,
            layout.storage_offset(),
            output_data.ptr() as *mut f32,
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
    f: CudaOp2Func,
) -> Result<Storage<T>> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(lhs_layout.len());
        f(
            lhs_data.ptr() as *const f32,
            lhs_layout.storage_offset(),
            lhs_layout.shape().as_ptr() as *const size_t,
            lhs_layout.stride().as_ptr() as *const size_t,
            lhs_layout.ndim(),
            rhs_data.ptr() as *mut f32,
            rhs_layout.storage_offset(),
            rhs_layout.shape().as_ptr() as *const size_t,
            rhs_layout.stride().as_ptr() as *const size_t,
            rhs_layout.ndim(),
            output_data.ptr() as *mut f32,
            lhs_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}
