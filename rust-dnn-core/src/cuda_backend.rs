use std::ffi::c_float;

use rust_dnn_cuda_kernel::basic::{
    cuda_add_double, cuda_add_float, cuda_add_uint32_t, cuda_contiguous_double,
    cuda_contiguous_float, cuda_div_double, cuda_div_float, cuda_div_uint32_t, cuda_mul_double,
    cuda_mul_float, cuda_mul_uint32_t, cuda_neg_double, cuda_neg_float, cuda_sub_double,
    cuda_sub_float, cuda_sub_uint32_t,
};
use rust_dnn_cuda_kernel::cuda::check_cuda_error;
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;
use rust_dnn_cuda_kernel::math::{
    cuda_pow_double, cuda_pow_float, cuda_sum_axis_double, cuda_sum_axis_float
};

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::Result;
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};
use libc::{c_double, size_t};

pub(crate) type CudaOp1FuncF32 =
    unsafe extern "C" fn(a: *const c_float, a_base_offset: size_t, b: *mut c_float, len: i32);

pub(crate) type CudaOp1FuncF64 =
    unsafe extern "C" fn(a: *const c_double, a_base_offset: size_t, b: *mut c_double, len: i32);

pub(crate) type CudaU32Op2Func = unsafe extern "C" fn(
    a: *const u32,
    a_base_offset: size_t,
    a_shape: *const size_t,
    a_strides: *const size_t,
    a_ndim: size_t,
    b: *const u32,
    b_base_offset: size_t,
    b_shape: *const size_t,
    b_strides: *const size_t,
    b_ndim: size_t,
    c: *mut u32,
    len: i32,
);

pub(crate) type CudaF32Op2Func = unsafe extern "C" fn(
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

pub(crate) type CudaF64Op2Func = unsafe extern "C" fn(
    a: *const c_double,
    a_base_offset: size_t,
    a_shape: *const size_t,
    a_strides: *const size_t,
    a_ndim: size_t,
    b: *const c_double,
    b_base_offset: size_t,
    b_shape: *const size_t,
    b_strides: *const size_t,
    b_ndim: size_t,
    c: *mut c_double,
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
            match T::dtype() {
                DType::F32 => {
                    cuda_sum_axis_float(
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
                }
                DType::F64 => {
                    cuda_sum_axis_double(
                        input_data.ptr() as *const f64,
                        input_layout.storage_offset(),
                        input_layout.shape().as_ptr() as *const size_t,
                        input_layout.stride().as_ptr() as *const size_t,
                        input_layout.ndim(),
                        output_data.ptr() as *mut f64,
                        output_layout.storage_offset(),
                        output_layout.shape().as_ptr() as *const size_t,
                        output_layout.stride().as_ptr() as *const size_t,
                        output_layout.ndim(),
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
        cuda_op2_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_add_uint32_t,
            cuda_add_float,
            cuda_add_double,
        )
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_sub_uint32_t,
            cuda_sub_float,
            cuda_sub_double,
        )
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_mul_uint32_t,
            cuda_mul_float,
            cuda_mul_double,
        )
    }

    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_div_uint32_t,
            cuda_div_float,
            cuda_div_double,
        )
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_neg_float, cuda_neg_double)
    }

    fn op_pow_scalar<T: Float>(
        storage: &Storage<T>,
        layout: &Layout,
        rhs: T,
    ) -> Result<Storage<T>> {
        let input_data = storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(layout.len());
            match T::dtype() {
                DType::F32 => {
                    cuda_pow_float(
                        input_data.ptr() as *const f32,
                        layout.storage_offset(),
                        output_data.ptr() as *mut f32,
                        rhs.as_f32(),
                        layout.len() as i32,
                    );
                }
                DType::F64 => {
                    cuda_pow_double(
                        input_data.ptr() as *const f64,
                        layout.storage_offset(),
                        output_data.ptr() as *mut f64,
                        rhs.as_f64(),
                        layout.len() as i32,
                    );
                }
                _ => panic!(),
            }
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }
}

pub(crate) fn cuda_op1_func_call<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f_f32: CudaOp1FuncF32,
    f_f64: CudaOp1FuncF64,
) -> Result<Storage<T>> {
    match T::dtype() {
        DType::F32 => cuda_op1_func_call_f32(storage, layout, f_f32),
        DType::F64 => cuda_op1_func_call_f64(storage, layout, f_f64),
        _ => panic!(),
    }
}

pub(crate) fn cuda_op1_func_call_f32<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f: CudaOp1FuncF32,
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

pub(crate) fn cuda_op1_func_call_f64<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f: CudaOp1FuncF64,
) -> Result<Storage<T>> {
    let input_data = storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(layout.len());
        f(
            input_data.ptr() as *const f64,
            layout.storage_offset(),
            output_data.ptr() as *mut f64,
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
    f_u32: CudaU32Op2Func,
    f_f32: CudaF32Op2Func,
    f_f64: CudaF64Op2Func,
) -> Result<Storage<T>> {
    match T::dtype() {
        DType::U32 => {
            cuda_u32_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, f_u32)
        }
        DType::F32 => {
            cuda_f32_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, f_f32)
        }
        DType::F64 => {
            cuda_f64_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, f_f64)
        }
    }
}

pub(crate) fn cuda_u32_op2_func_call<T: Num>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaU32Op2Func,
) -> Result<Storage<T>> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(lhs_layout.len());
        f(
            lhs_data.ptr() as *const u32,
            lhs_layout.storage_offset(),
            lhs_layout.shape().as_ptr() as *const size_t,
            lhs_layout.stride().as_ptr() as *const size_t,
            lhs_layout.ndim(),
            rhs_data.ptr() as *mut u32,
            rhs_layout.storage_offset(),
            rhs_layout.shape().as_ptr() as *const size_t,
            rhs_layout.stride().as_ptr() as *const size_t,
            rhs_layout.ndim(),
            output_data.ptr() as *mut u32,
            lhs_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

pub(crate) fn cuda_f32_op2_func_call<T: Num>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaF32Op2Func,
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

pub(crate) fn cuda_f64_op2_func_call<T: Num>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaF64Op2Func,
) -> Result<Storage<T>> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(lhs_layout.len());
        f(
            lhs_data.ptr() as *const f64,
            lhs_layout.storage_offset(),
            lhs_layout.shape().as_ptr() as *const size_t,
            lhs_layout.stride().as_ptr() as *const size_t,
            lhs_layout.ndim(),
            rhs_data.ptr() as *mut f64,
            rhs_layout.storage_offset(),
            rhs_layout.shape().as_ptr() as *const size_t,
            rhs_layout.stride().as_ptr() as *const size_t,
            rhs_layout.ndim(),
            output_data.ptr() as *mut f64,
            lhs_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}
