use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaBackend;
use crate::{backend::Backend, cpu_backend::CpuBackend};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceInfo {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
}

impl DeviceInfo {
    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Cpu => true,
            #[cfg(feature = "cuda")]
            Self::Cuda => false,
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "cuda")]
            Self::Cuda => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Device<B: Backend> {
    info: DeviceInfo,
    _marker: PhantomData<B>,
}

impl<B: Backend> Device<B> {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    pub fn is_cpu(&self) -> bool {
        self.info.is_cpu()
    }

    pub fn is_cuda(&self) -> bool {
        self.info.is_cuda()
    }
}

impl Device<CpuBackend> {
    pub fn get_cpu_device() -> Self {
        Self {
            info: DeviceInfo::Cpu,
            _marker: PhantomData,
        }
    }
}

#[cfg(feature = "cuda")]
impl Device<CudaBackend> {
    pub fn get_cuda_device() -> Self {
        Self {
            info: DeviceInfo::Cuda,
            _marker: PhantomData,
        }
    }
}
