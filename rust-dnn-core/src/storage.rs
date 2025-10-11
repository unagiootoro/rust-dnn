#[cfg(feature = "cuda")]
use crate::error::Error;
use crate::error::Result;
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

    #[cfg(feature = "cuda")]
    pub fn get_cuda_storage(&self) -> Result<&GPUBuffer<T>> {
        match self {
            Self::CudaStorage(cuda_storage) => Ok(cuda_storage),
            Self::CpuStorage(_) => Err(Error::DeviceError {
                msg: "Failed to get CudaStorage due to CpuStorage.".to_string(),
            }),
        }
    }
}
