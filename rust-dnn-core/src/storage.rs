#[cfg(feature = "cuda")]
use crate::error::Error;
use crate::{dtype::DType, error::Result};
use std::ops::Range;

#[cfg(feature = "cuda")]
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;

use crate::num::Num;

pub enum Storage<T: Num> {
    CpuStorage(Vec<T>),
    #[cfg(feature = "cuda")]
    CudaStorage(GPUBuffer<T>),
}

impl<T: Num> Storage<T> {
    pub fn to_vec_range(&self, range: Range<usize>) -> Vec<T> {
        match self {
            Self::CpuStorage(cpu_storage) => cpu_storage[range].to_vec(),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(cuda_storage) => cuda_storage.to_vec()[range].to_vec(),
        }
    }

    pub fn get_cpu_storage(&self) -> Result<&Vec<T>> {
        match self {
            Self::CpuStorage(cpu_storage) => Ok(cpu_storage),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CpuStorage due to CudaStorage.".to_string(),
            }),
        }
    }

    pub fn get_cpu_storage_mut(&mut self) -> Result<&mut Vec<T>> {
        match self {
            Self::CpuStorage(cpu_storage) => Ok(cpu_storage),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CpuStorage due to CudaStorage.".to_string(),
            }),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn get_cuda_storage(&self) -> Result<&GPUBuffer<T>> {
        match self {
            Self::CudaStorage(cuda_storage) => Ok(cuda_storage),
            Self::CpuStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CudaStorage due to CpuStorage.".to_string(),
            }),
        }
    }

    pub fn to_dtype<T2: Num>(&self) -> Storage<T2> {
        match self {
            Self::CpuStorage(cpu_storage) => Storage::CpuStorage(convert_vec::<T, T2>(cpu_storage)),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(cuda_storage) => todo!(),
        }
    }
}

fn convert_vec<T1: Num, T2: Num>(v: &Vec<T1>) -> Vec<T2> {
    match T2::dtype() {
        DType::U32 => {
            let v2 = v.iter().map(|n| n.as_u32()).collect();
            unsafe { std::mem::transmute::<Vec<u32>, Vec<T2>>(v2) }
        }
        DType::F32 => {
            let v2 = v.iter().map(|n| n.as_f32()).collect();
            unsafe { std::mem::transmute::<Vec<f32>, Vec<T2>>(v2) }
        }
        DType::F64 => {
            let v2 = v.iter().map(|n| n.as_f64()).collect();
            unsafe { std::mem::transmute::<Vec<f64>, Vec<T2>>(v2) }
        }
    }
}
