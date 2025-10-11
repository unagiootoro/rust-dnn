use crate::raw_gpu_buffer::RawGpuBuffer;

pub struct GpuMemoryBlock {
    pub buffer: RawGpuBuffer,
    pub locked: bool,
}

impl GpuMemoryBlock {
    pub fn new(buffer: RawGpuBuffer) -> GpuMemoryBlock {
        Self { buffer, locked: false }
    }

    pub fn unlock(&mut self) {
        self.locked = false;
    }
}
