use std::marker::PhantomData;

use crate::{backend::Backend, cpu_backend::CpuBackend};

#[derive(Debug, Clone, Copy)]
pub enum DeviceInfo {
    Cpu,
}

#[derive(Debug, Clone, Copy)]
pub struct Device<B: Backend> {
    info: DeviceInfo,
    _marker: PhantomData<B>,
}

impl<B: Backend> Device<B> {
    pub fn info(&self) -> &DeviceInfo {
        &self.info
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
