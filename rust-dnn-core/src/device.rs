use std::marker::PhantomData;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaBackend;
use crate::{backend::Backend, cpu_backend::CpuBackend, wgpu_backend::WgpuBackend};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceInfo {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda,
    Wgpu,
}

impl DeviceInfo {
    pub fn is_cpu(&self) -> bool {
        match self {
            Self::Cpu => true,
            #[cfg(feature = "cuda")]
            Self::Cuda => false,
            Self::Wgpu => false,
        }
    }

    pub fn is_cuda(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "cuda")]
            Self::Cuda => true,
            Self::Wgpu => false,
        }
    }

    pub fn is_wgpu(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "cuda")]
            Self::Cuda => false,
            Self::Wgpu => true,
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

    pub fn is_wgpu(&self) -> bool {
        self.info.is_wgpu()
    }

    pub unsafe fn reinterpret_cast<B2: Backend>(self) -> Device<B2> {
        unsafe { std::mem::transmute::<Self, Device<B2>>(self) }
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

impl Device<WgpuBackend> {
    pub fn get_wgpu_device() -> Self {
        Self {
            info: DeviceInfo::Wgpu,
            _marker: PhantomData,
        }
    }
}
