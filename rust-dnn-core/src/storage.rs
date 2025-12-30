use crate::error::Error;
use crate::{dtype::DType, error::Result};
use std::ops::Range;

#[cfg(feature = "cuda")]
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;

use rust_dnn_wgpu::wgpu_buffer::WgpuBuffer;

use crate::num::Num;

pub enum Storage<T: Num> {
    CpuStorage(Vec<T>),
    #[cfg(feature = "cuda")]
    CudaStorage(GPUBuffer<T>),
    WgpuStorage(WgpuBuffer),
}

impl<T: Num> Storage<T> {
    pub fn to_vec_range(&self, range: Range<usize>) -> Vec<T> {
        match self {
            Self::CpuStorage(cpu_storage) => cpu_storage[range].to_vec(),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(cuda_storage) => cuda_storage.to_vec()[range].to_vec(),
            Self::WgpuStorage(wgpu_storage) => {
                match T::dtype() {
                    DType::U32 => {
                        let v = wgpu_storage.to_vec_u32()[range].to_vec();
                        unsafe { std::mem::transmute::<Vec<u32>, Vec<T>>(v) }
                    },
                    DType::F32 => {
                        let v = wgpu_storage.to_vec_f32()[range].to_vec();
                        unsafe { std::mem::transmute::<Vec<f32>, Vec<T>>(v) }
                    },
                    _ => todo!()
                }
            }
        }
    }

    pub fn get_cpu_storage(&self) -> Result<&Vec<T>> {
        match self {
            Self::CpuStorage(cpu_storage) => Ok(cpu_storage),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CpuStorage due to CudaStorage.".to_string(),
            }),
            Self::WgpuStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CpuStorage due to WgpuStorage.".to_string(),
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
            Self::WgpuStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CpuStorage due to WgpuStorage.".to_string(),
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

    pub fn get_wgpu_storage(&self) -> Result<&WgpuBuffer> {
        match self {
            Self::CpuStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get WgpuStorage due to CpuStorage.".to_string(),
            }),
            #[cfg(feature = "cuda")]
            Self::CudaStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get WgpuStorage due to CudaStorage.".to_string(),
            }),
            Self::WgpuStorage(wgpu_storage) => Ok(wgpu_storage),
        }
    }
}
