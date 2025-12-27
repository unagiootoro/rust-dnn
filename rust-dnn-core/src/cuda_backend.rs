use std::ffi::{c_float, c_void};
use std::ptr;

use libc::c_int;
use rust_dnn_cuda_kernel::basic::*;
use rust_dnn_cuda_kernel::clayout::{CLayout, MAX_NDIM, NDimArray};
use rust_dnn_cuda_kernel::cublas::{
    CUBLAS_HANDLE, CUBLAS_OP_T, cublasCreate_v2, cublasSgemm_v2, cublasSgemmStridedBatched,
};
use rust_dnn_cuda_kernel::cuda::check_cuda_error;
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;
use rust_dnn_cuda_kernel::math::*;
use rust_dnn_cuda_kernel::nn::{cuda_col2im, cuda_im2col};

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};

pub(crate) type CudaOp1Func<T> =
    unsafe fn(a: *const T, a_storage_offset: usize, b: *mut T, len: i32);

pub(crate) type CudaOp2Func<T1, T2> = unsafe fn(
    a: *const T1,
    a_layout: CLayout,
    b: *const T1,
    b_layout: CLayout,
    c: *mut T2,
    len: i32,
);

pub(crate) type CudaOp2AssignFunc<T> =
    unsafe fn(a: *mut T, a_layout: CLayout, b: *const T, b_layout: CLayout, len: i32);

pub(crate) type CudaReduceFunc<T> = unsafe fn(a: *const T, a_layout: CLayout, b: *mut T, len: i32);

pub(crate) type CudaReduceAxisFunc<T> =
    unsafe fn(a: *const T, a_layout: CLayout, b: *mut T, b_layout: CLayout, axis: usize, len: i32);

pub(crate) type CudaReduceCmpAxisIndexFunc<T> = unsafe fn(
    a: *const T,
    a_layout: CLayout,
    b: *mut u32,
    b_layout: CLayout,
    axis: usize,
    len: i32,
);

#[derive(Debug, Clone, Copy)]
pub struct CudaBackend;

impl Backend for CudaBackend {
    fn convert_dtype<T1: Num, T2: Num>(
        storage: &Storage<T1>,
        layout: &Layout,
    ) -> Result<Storage<T2>> {
        let input_data = storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T2>::new(layout.len());
            cuda_convert(
                T1::dtype() as i32,
                T2::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(&layout)?,
                output_data.ptr(),
                layout.len() as i32,
            );
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn contiguous<T: Num>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        let input_data = storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(layout.len());
            cuda_contiguous(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(&layout)?,
                output_data.ptr(),
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
        cuda_reduce_axis_func_call(
            input_storage,
            input_layout,
            output_layout,
            axis,
            cuda_sum_axis,
        )
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

    fn eq<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        cuda_op2_func_call::<T, u32>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_eq)
    }

    fn lt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_lt)
    }

    fn le<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_le)
    }

    fn gt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_gt)
    }

    fn ge<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_ge)
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_neg)
    }

    fn copy<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        cuda_op2_assign_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_copy)
    }

    fn pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        cuda_op2_func_call(lhs_storage, rhs_storage, lhs_layout, rhs_layout, cuda_pow)
    }

    fn sin<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_sin)
    }

    fn cos<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_cos)
    }

    fn tanh<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_tanh)
    }

    fn sqrt<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_sqrt)
    }

    fn ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_ln)
    }

    fn matmul<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        let lhs_rows = lhs_layout.shape()[0];
        let rhs_cols = rhs_layout.shape()[1];

        let lhs_data = lhs_storage.get_cuda_storage()?;
        let rhs_data = rhs_storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(lhs_rows * rhs_cols);
            cuda_matmul(
                T::dtype() as i32,
                lhs_data.ptr(),
                layout_to_clayout(lhs_layout)?,
                rhs_data.ptr(),
                layout_to_clayout(rhs_layout)?,
                output_data.ptr(),
                (lhs_rows * rhs_cols) as i32,
            );
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn is_cublas_supported() -> bool {
        true
    }

    fn cublas_sgemm<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        let a_rows = lhs_layout.shape()[0];
        let a_cols = lhs_layout.shape()[1];
        let b_cols = rhs_layout.shape()[1];

        let a_data = lhs_storage.get_cuda_storage()?;
        let b_data = rhs_storage.get_cuda_storage()?;

        let output_len = a_rows * b_cols;
        let output_data = unsafe { GPUBuffer::<T>::new(output_len) };

        let m = a_rows;
        let n = b_cols;
        let k = a_cols;

        let handle = if let Some(handle) = CUBLAS_HANDLE.with(|handle| *handle.borrow()) {
            handle
        } else {
            let mut handle: *mut c_void = ptr::null_mut();
            unsafe {
                cublasCreate_v2(&mut handle);
            };
            CUBLAS_HANDLE.with(|h| *h.borrow_mut() = Some(handle));
            handle
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        unsafe {
            cublasSgemm_v2(
                handle,
                CUBLAS_OP_T, // ← A を転置（行優先→列優先の意味）
                CUBLAS_OP_T, // ← B も転置
                m as c_int,  // rows of op(A) = a_rows
                n as c_int,  // cols of op(B) = b_cols
                k as c_int,  // cols of op(A) = a_cols
                &alpha,
                a_data.ptr() as *const f32,
                k as c_int, // lda = cols of A in row-major
                b_data.ptr() as *const f32,
                n as c_int, // ldb = cols of B in row-major
                &beta,
                output_data.ptr() as *mut f32,
                m as c_int, // ldc = cols of C in row-major
            );
        }

        // let mut out = Tensor::new(
        //     None,
        //     vec![b_cols, a_rows],
        //     None,
        //     Rc::new(RefCell::new(TensorStorage::GpuStorage(result))),
        //     false,
        // );
        // let out = out.reversed_axes();
        // let mut out = out.contiguous();
        // out.set_requires_grad(a.is_requires_grad() || b.is_requires_grad());
        // out

        Ok(Storage::CudaStorage(output_data))
    }

    fn cublas_sgemm_batched<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        batch_size: usize,
    ) -> Result<Storage<T>> {
        let a_rows = lhs_layout.shape()[0];
        let a_cols = lhs_layout.shape()[1];
        let b_cols = rhs_layout.shape()[1];

        let a_data = lhs_storage.get_cuda_storage()?;
        let b_data = rhs_storage.get_cuda_storage()?;

        let output_len = batch_size * a_rows * b_cols;
        let output_data = unsafe { GPUBuffer::<T>::new(output_len) };

        let m = a_rows;
        let n = b_cols;
        let k = a_cols;

        let handle = if let Some(handle) = CUBLAS_HANDLE.with(|handle| *handle.borrow()) {
            handle
        } else {
            let mut handle: *mut c_void = ptr::null_mut();
            unsafe {
                cublasCreate_v2(&mut handle);
            };
            CUBLAS_HANDLE.with(|h| *h.borrow_mut() = Some(handle));
            handle
        };

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let stride_a: i64 = (m * k) as i64; // A の1バッチ分の要素数
        let stride_b: i64 = (k * n) as i64; // B の1バッチ分の要素数
        let stride_c: i64 = (m * n) as i64; // C の1バッチ分の要素数

        unsafe {
            let a_ptr = (a_data.ptr() as *const f32).offset(lhs_layout.storage_offset() as isize);
            let b_ptr = (b_data.ptr() as *const f32).offset(rhs_layout.storage_offset() as isize);
            cublasSgemmStridedBatched(
                handle,
                CUBLAS_OP_T, // ← A を転置（行優先→列優先の意味）
                CUBLAS_OP_T, // ← B も転置
                m as c_int,  // rows of op(A) = a_rows
                n as c_int,  // cols of op(B) = b_cols
                k as c_int,  // cols of op(A) = a_cols
                &alpha,
                a_ptr,
                k as c_int, // lda = cols of A in row-major
                stride_a,
                b_ptr,
                n as c_int, // ldb = cols of B in row-major
                stride_b,
                &beta,
                output_data.ptr() as *mut f32,
                m as c_int, // ldc = cols of C in row-major
                stride_c,
                batch_size as i32,
            );
        }

        // let out = Tensor::new(
        //     None,
        //     vec![batch_size, b_cols, a_rows],
        //     None,
        //     Rc::new(RefCell::new(TensorStorage::GpuStorage(result))),
        //     a.is_requires_grad() || b.is_requires_grad(),
        // );
        // let out = out.permuted_axes(vec![0, 2, 1]);
        // let out = out.contiguous();

        // let mut out_shape = batch_a_shape;
        // out_shape.push(a.shape()[1]);
        // out_shape.push(b.shape()[2]);
        // let mut out = out.reshape(out_shape);
        // out.set_requires_grad(a.is_requires_grad() || b.is_requires_grad());
        // out

        Ok(Storage::CudaStorage(output_data))
    }

    fn gather<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(index_layout.len());
            cuda_gather(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                output_data.ptr(),
                axis,
                index_layout.len() as i32,
            );
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn scatter<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let src_data = src_storage.get_cuda_storage()?;

        let mut len = 1;
        for (i, dim) in index_layout.shape().iter().enumerate() {
            if i != axis {
                len *= *dim;
            }
        }

        unsafe {
            cuda_scatter(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                src_data.ptr(),
                layout_to_clayout(src_layout)?,
                axis,
                len as i32,
            );
            check_cuda_error();
        };
        Ok(())
    }

    fn scatter_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let src_data = src_storage.get_cuda_storage()?;

        let mut len = 1;
        for (i, dim) in index_layout.shape().iter().enumerate() {
            if i != axis {
                len *= *dim;
            }
        }

        unsafe {
            cuda_scatter_add(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                src_data.ptr(),
                layout_to_clayout(src_layout)?,
                axis,
                len as i32,
            );
            check_cuda_error();
        };
        Ok(())
    }

    fn index_select<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let output_data = unsafe {
            let output_data = GPUBuffer::<T>::new(output_layout.len());
            cuda_index_select(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                output_data.ptr(),
                layout_to_clayout(output_layout)?,
                axis,
                output_layout.len() as i32,
            );
            check_cuda_error();
            output_data
        };
        Ok(Storage::CudaStorage(output_data))
    }

    fn index_copy<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let src_data = src_storage.get_cuda_storage()?;

        let mut len = 1;
        for (i, dim) in src_layout.shape().iter().enumerate() {
            if i != axis {
                len *= *dim;
            }
        }

        unsafe {
            cuda_index_copy(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                src_data.ptr(),
                layout_to_clayout(src_layout)?,
                axis,
                len as i32,
            );
            check_cuda_error();
        };
        Ok(())
    }

    fn index_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        let input_data = input_storage.get_cuda_storage()?;
        let index_data = index_storage.get_cuda_storage()?;
        let src_data = src_storage.get_cuda_storage()?;

        let mut len = 1;
        for (i, dim) in src_layout.shape().iter().enumerate() {
            if i != axis {
                len *= *dim;
            }
        }

        unsafe {
            cuda_index_add(
                T::dtype() as i32,
                input_data.ptr(),
                layout_to_clayout(input_layout)?,
                index_data.ptr() as *const u32,
                layout_to_clayout(index_layout)?,
                src_data.ptr(),
                layout_to_clayout(src_layout)?,
                axis,
                len as i32,
            );
            check_cuda_error();
        };
        Ok(())
    }

    fn sum<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        cuda_reduce_func_call(input_storage, input_layout, cuda_sum)
    }

    fn max<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        cuda_reduce_func_call(input_storage, input_layout, cuda_max)
    }

    fn max_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        cuda_reduce_axis_func_call(
            input_storage,
            input_layout,
            output_layout,
            axis,
            cuda_max_axis,
        )
    }

    fn op_add_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        cuda_op2_assign_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_add_assign,
        )
    }

    fn op_sub_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        cuda_op2_assign_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_sub_assign,
        )
    }

    fn op_mul_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        cuda_op2_assign_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_mul_assign,
        )
    }

    fn op_div_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        cuda_op2_assign_func_call(
            lhs_storage,
            rhs_storage,
            lhs_layout,
            rhs_layout,
            cuda_div_assign,
        )
    }

    fn exp<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        cuda_op1_func_call(storage, layout, cuda_exp)
    }

    fn argmax_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<u32>> {
        cuda_reduce_cmp_axis_index_func_call(
            input_storage,
            input_layout,
            output_layout,
            axis,
            cuda_argmax_axis,
        )
    }

    fn im2col<T: Float>(
        img_storage: &Storage<T>,
        col_storage: &mut Storage<T>,
        ch: usize,
        img_h: usize,
        img_w: usize,
        out_h: usize,
        out_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<()> {
        let img_data = img_storage.get_cuda_storage()?;
        let col_data = col_storage.get_cuda_storage()?;
        unsafe {
            cuda_im2col(
                T::dtype() as i32,
                img_data.ptr(),
                img_data.len() as i32,
                col_data.ptr(),
                col_data.len() as i32,
                0,
                ch,
                img_h,
                img_w,
                out_h,
                out_w,
                fil_h,
                fil_w,
                stride_h,
                stride_w,
            );
        }
        Ok(())
    }

    fn col2im<T: Float>(
        col_storage: &Storage<T>,
        img_storage: &mut Storage<T>,
        img_shape: &Vec<usize>,
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
    ) -> Result<()> {
        let col_data = col_storage.get_cuda_storage()?;
        let img_data = img_storage.get_cuda_storage()?;
        unsafe {
            cuda_col2im(
                T::dtype() as i32,
                col_data.ptr(),
                col_data.len() as i32,
                img_data.ptr(),
                img_data.len() as i32,
                bsize,
                ch,
                img_h,
                img_w,
                out_h,
                out_w,
                fil_h,
                fil_w,
                stride_h,
                stride_w,
            );
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! define_cuda_op1_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(a: *const T, a_storage_offset: usize, b: *mut T, len: i32) {
            unsafe {
                match T::dtype() {
                    DType::F32 => {
                        $fn_name_f32(a as *const f32, a_storage_offset, b as *mut f32, len)
                    }
                    DType::F64 => {
                        $fn_name_f64(a as *const f64, a_storage_offset, b as *mut f64, len)
                    }
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
macro_rules! define_cuda_op2_u32_func {
    ($fn_name: ident, $fn_name_u32: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(
            a: *const T,
            a_layout: CLayout,
            b: *const T,
            b_layout: CLayout,
            c: *mut u32,
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
                        c as *mut u32,
                        len,
                    ),
                    DType::F64 => $fn_name_f64(
                        a as *const f64,
                        a_layout,
                        b as *const f64,
                        b_layout,
                        c as *mut u32,
                        len,
                    ),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_cuda_op2_assign_func {
    ($fn_name: ident, $fn_name_u32: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(
            a: *mut T,
            a_layout: CLayout,
            b: *const T,
            b_layout: CLayout,
            len: i32,
        ) {
            unsafe {
                match T::dtype() {
                    DType::U32 => {
                        $fn_name_u32(a as *mut u32, a_layout, b as *const u32, b_layout, len)
                    }
                    DType::F32 => {
                        $fn_name_f32(a as *mut f32, a_layout, b as *const f32, b_layout, len)
                    }
                    DType::F64 => {
                        $fn_name_f64(a as *mut f64, a_layout, b as *const f64, b_layout, len)
                    }
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

#[macro_export]
macro_rules! define_cuda_reduce_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(a: *const T, a_layout: CLayout, b: *mut T, len: i32) {
            unsafe {
                match T::dtype() {
                    DType::F32 => $fn_name_f32(a as *const f32, a_layout, b as *mut f32, len),
                    DType::F64 => $fn_name_f64(a as *const f64, a_layout, b as *mut f64, len),
                    _ => panic!(),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_cuda_reduce_axis_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(
            a: *const T,
            a_layout: CLayout,
            b: *mut T,
            b_layout: CLayout,
            axis: usize,
            len: i32,
        ) {
            unsafe {
                match T::dtype() {
                    DType::F32 => $fn_name_f32(
                        a as *const f32,
                        a_layout,
                        b as *mut f32,
                        b_layout,
                        axis,
                        len,
                    ),
                    DType::F64 => $fn_name_f64(
                        a as *const f64,
                        a_layout,
                        b as *mut f64,
                        b_layout,
                        axis,
                        len,
                    ),
                    _ => panic!(),
                }
            }
        }
    };
}

#[macro_export]
macro_rules! define_cuda_reduce_cmp_axis_index_func {
    ($fn_name: ident, $fn_name_f32: ident, $fn_name_f64: ident) => {
        unsafe fn $fn_name<T: Num>(
            a: *const T,
            a_layout: CLayout,
            b: *mut u32,
            b_layout: CLayout,
            axis: usize,
            len: i32,
        ) {
            unsafe {
                match T::dtype() {
                    DType::F32 => $fn_name_f32(
                        a as *const f32,
                        a_layout,
                        b as *mut u32,
                        b_layout,
                        axis,
                        len,
                    ),
                    DType::F64 => $fn_name_f64(
                        a as *const f64,
                        a_layout,
                        b as *mut u32,
                        b_layout,
                        axis,
                        len,
                    ),
                    _ => panic!(),
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

define_cuda_op2_assign_func!(
    cuda_copy,
    cuda_copy_uint32_t,
    cuda_copy_float,
    cuda_copy_double
);

define_cuda_op2_assign_func!(
    cuda_add_assign,
    cuda_add_assign_uint32_t,
    cuda_add_assign_float,
    cuda_add_assign_double
);

define_cuda_op2_assign_func!(
    cuda_sub_assign,
    cuda_sub_assign_uint32_t,
    cuda_sub_assign_float,
    cuda_sub_assign_double
);

define_cuda_op2_assign_func!(
    cuda_mul_assign,
    cuda_mul_assign_uint32_t,
    cuda_mul_assign_float,
    cuda_mul_assign_double
);

define_cuda_op2_assign_func!(
    cuda_div_assign,
    cuda_div_assign_uint32_t,
    cuda_div_assign_float,
    cuda_div_assign_double
);

define_cuda_op2_u32_func!(cuda_lt, cuda_lt_uint32_t, cuda_lt_float, cuda_lt_double);
define_cuda_op2_u32_func!(cuda_le, cuda_le_uint32_t, cuda_le_float, cuda_le_double);
define_cuda_op2_u32_func!(cuda_gt, cuda_gt_uint32_t, cuda_gt_float, cuda_gt_double);
define_cuda_op2_u32_func!(cuda_ge, cuda_ge_uint32_t, cuda_ge_float, cuda_ge_double);
define_cuda_op2_u32_func!(cuda_eq, cuda_eq_uint32_t, cuda_eq_float, cuda_eq_double);

define_cuda_float_op2_func!(cuda_pow, cuda_pow_float, cuda_pow_double);
define_cuda_op1_func!(cuda_exp, cuda_exp_float, cuda_exp_double);
define_cuda_op1_func!(cuda_ln, cuda_ln_float, cuda_ln_double);
define_cuda_op1_func!(cuda_sqrt, cuda_sqrt_float, cuda_sqrt_double);
define_cuda_op1_func!(cuda_sin, cuda_sin_float, cuda_sin_double);
define_cuda_op1_func!(cuda_cos, cuda_cos_float, cuda_cos_double);
define_cuda_op1_func!(cuda_tanh, cuda_tanh_float, cuda_tanh_double);

define_cuda_reduce_func!(cuda_sum, cuda_sum_float, cuda_sum_double);
define_cuda_reduce_func!(cuda_max, cuda_max_float, cuda_max_double);

define_cuda_reduce_axis_func!(cuda_sum_axis, cuda_sum_axis_float, cuda_sum_axis_double);
define_cuda_reduce_axis_func!(cuda_max_axis, cuda_max_axis_float, cuda_max_axis_double);

define_cuda_reduce_cmp_axis_index_func!(
    cuda_argmax_axis,
    cuda_argmax_axis_float,
    cuda_argmax_axis_double
);

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

pub(crate) fn cuda_op2_func_call<T1: Num, T2: Num>(
    lhs_storage: &Storage<T1>,
    rhs_storage: &Storage<T1>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaOp2Func<T1, T2>,
) -> Result<Storage<T2>> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let lhs_clayout = layout_to_clayout(lhs_layout)?;
    let rhs_clayout = layout_to_clayout(rhs_layout)?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T2>::new(lhs_layout.len());
        f(
            lhs_data.ptr() as *const T1,
            lhs_clayout,
            rhs_data.ptr() as *mut T1,
            rhs_clayout,
            output_data.ptr() as *mut T2,
            lhs_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

pub(crate) fn cuda_op2_assign_func_call<T: Num>(
    lhs_storage: &mut Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: CudaOp2AssignFunc<T>,
) -> Result<()> {
    let lhs_data = lhs_storage.get_cuda_storage()?;
    let rhs_data = rhs_storage.get_cuda_storage()?;
    let lhs_clayout = layout_to_clayout(lhs_layout)?;
    let rhs_clayout = layout_to_clayout(rhs_layout)?;
    unsafe {
        f(
            lhs_data.ptr() as *mut T,
            lhs_clayout,
            rhs_data.ptr() as *const T,
            rhs_clayout,
            lhs_layout.len() as i32,
        );
    }
    check_cuda_error();
    Ok(())
}

pub(crate) fn cuda_reduce_func_call<T: Num>(
    storage: &Storage<T>,
    layout: &Layout,
    f: CudaReduceFunc<T>,
) -> Result<Storage<T>> {
    let input_data = storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(layout.len());
        f(
            input_data.ptr() as *const T,
            layout_to_clayout(layout)?,
            output_data.ptr() as *mut T,
            layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

pub(crate) fn cuda_reduce_axis_func_call<T: Num>(
    input_storage: &Storage<T>,
    input_layout: &Layout,
    output_layout: &Layout,
    axis: usize,
    f: CudaReduceAxisFunc<T>,
) -> Result<Storage<T>> {
    let input_data = input_storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<T>::new(output_layout.len());
        f(
            input_data.ptr() as *const T,
            layout_to_clayout(input_layout)?,
            output_data.ptr() as *mut T,
            layout_to_clayout(output_layout)?,
            axis,
            output_layout.len() as i32,
        );
        check_cuda_error();
        output_data
    };
    Ok(Storage::CudaStorage(output_data))
}

pub(crate) fn cuda_reduce_cmp_axis_index_func_call<T: Num>(
    input_storage: &Storage<T>,
    input_layout: &Layout,
    output_layout: &Layout,
    axis: usize,
    f: CudaReduceCmpAxisIndexFunc<T>,
) -> Result<Storage<u32>> {
    let input_data = input_storage.get_cuda_storage()?;
    let output_data = unsafe {
        let output_data = GPUBuffer::<u32>::new(output_layout.len());
        f(
            input_data.ptr() as *const T,
            layout_to_clayout(input_layout)?,
            output_data.ptr() as *mut u32,
            layout_to_clayout(output_layout)?,
            axis,
            input_layout.len() as i32,
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

fn slice_to_ndimarray(slice: &[usize]) -> Result<NDimArray> {
    if slice.len() > MAX_NDIM {
        return Err(Error::ArgumentsError {
            msg: format!("Invalid ndim({})", slice.len()).to_string(),
        });
    }
    let mut array = [0; MAX_NDIM];
    for (i, dim) in slice.iter().enumerate() {
        array[i] = *dim;
    }
    let ndimarray = NDimArray {
        data: array,
        ndim: slice.len(),
    };
    Ok(ndimarray)
}
