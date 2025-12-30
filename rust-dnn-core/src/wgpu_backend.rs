use rust_dnn_wgpu::layout::MAX_NDIM;
use rust_dnn_wgpu::shader_type::ShaderType;
use rust_dnn_wgpu::{wgpu_add, wgpu_sub};
use rust_dnn_wgpu::wgpu_buffer::WgpuBuffer;

use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};

#[derive(Debug, Clone, Copy)]
pub struct WgpuBackend;

impl Backend for WgpuBackend {
    fn convert_dtype<T1: Num, T2: Num>(
        storage: &Storage<T1>,
        layout: &Layout,
    ) -> Result<Storage<T2>> {
        todo!()
    }

    fn contiguous<T: Num>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn sum_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn op_add<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        let output_data = match T::dtype() {
            DType::U32 => {
                let lhs_data = lhs_storage.get_wgpu_storage().unwrap();
                let rhs_data = rhs_storage.get_wgpu_storage().unwrap();
                let output_data = WgpuBuffer::zeros_u32(lhs_layout.len());
                wgpu_add(
                    &lhs_data,
                    layout_to_wgpu_layout(lhs_layout),
                    &rhs_data,
                    layout_to_wgpu_layout(rhs_layout),
                    &output_data,
                    lhs_layout.len() as u32,
                    ShaderType::Op2U32,
                );
                output_data
            }
            DType::F32 => {
                let lhs_data = lhs_storage.get_wgpu_storage().unwrap();
                let rhs_data = rhs_storage.get_wgpu_storage().unwrap();
                let output_data = WgpuBuffer::zeros_f32(lhs_layout.len());
                wgpu_add(
                    &lhs_data,
                    layout_to_wgpu_layout(lhs_layout),
                    &rhs_data,
                    layout_to_wgpu_layout(rhs_layout),
                    &output_data,
                    lhs_layout.len() as u32,
                    ShaderType::Op2F32,
                );
                output_data
            }
            DType::F64 => todo!()
        };

        Ok(Storage::WgpuStorage(output_data))
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        let output_data = match T::dtype() {
            DType::U32 => {
                let lhs_data = lhs_storage.get_wgpu_storage().unwrap();
                let rhs_data = rhs_storage.get_wgpu_storage().unwrap();
                let output_data = WgpuBuffer::zeros_u32(lhs_layout.len());
                wgpu_sub(
                    &lhs_data,
                    layout_to_wgpu_layout(lhs_layout),
                    &rhs_data,
                    layout_to_wgpu_layout(rhs_layout),
                    &output_data,
                    lhs_layout.len() as u32,
                    ShaderType::Op2U32,
                );
                output_data
            }
            DType::F32 => {
                let lhs_data = lhs_storage.get_wgpu_storage().unwrap();
                let rhs_data = rhs_storage.get_wgpu_storage().unwrap();
                let output_data = WgpuBuffer::zeros_f32(lhs_layout.len());
                wgpu_sub(
                    &lhs_data,
                    layout_to_wgpu_layout(lhs_layout),
                    &rhs_data,
                    layout_to_wgpu_layout(rhs_layout),
                    &output_data,
                    lhs_layout.len() as u32,
                    ShaderType::Op2F32,
                );
                output_data
            }
            DType::F64 => todo!()
        };

        Ok(Storage::WgpuStorage(output_data))
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn eq<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        todo!()
    }

    fn lt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        todo!()
    }

    fn le<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        todo!()
    }

    fn gt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        todo!()
    }

    fn ge<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        todo!()
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn copy<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn sin<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn cos<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn tanh<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn sqrt<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn matmul<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn is_cublas_supported() -> bool {
        false
    }

    fn cublas_sgemm<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn cublas_sgemm_batched<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        batch_size: usize,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn gather<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn index_select<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        todo!()
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
        todo!()
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
        todo!()
    }

    fn sum<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn max<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn max_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        todo!()
    }

    fn op_add_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn op_sub_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn op_mul_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn op_div_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        todo!()
    }

    fn exp<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        todo!()
    }

    fn argmax_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<u32>> {
        todo!()
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
        todo!()
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
        todo!()
    }
}

fn layout_to_wgpu_layout(layout: &Layout) -> rust_dnn_wgpu::layout::Layout {
    // rust_dnn_wgpu::layout::Layout::new(
    //     layout.shape().to_vec(),
    //     layout.stride().to_vec(),
    //     layout.storage_offset(),
    // )

    if layout.ndim() > MAX_NDIM {
        panic!("Invalid layout ndim({})", layout.ndim());
    }
    let mut shape = [0u32; MAX_NDIM];
    for (i, dim) in layout.shape().iter().enumerate() {
        shape[i] = *dim as u32;
    }
    let mut stride = [0u32; MAX_NDIM];
    for (i, dim) in layout.stride().iter().enumerate() {
        stride[i] = *dim as u32;
    }
    let wgpu_layout = rust_dnn_wgpu::layout::Layout::new(
        shape,
        stride,
        layout.ndim() as u32,
        layout.len() as u32,
        layout.storage_offset() as u32,
    );
    wgpu_layout
}
