pub const MAX_NDIM: usize = 8;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Layout {
    shape: [u32; MAX_NDIM],
    stride: [u32; MAX_NDIM],
    ndim: u32,
    len: u32,
    storage_offset: u32,
    _padding: u32,
}

impl Layout {
    pub fn new(shape: [u32; MAX_NDIM], stride: [u32; MAX_NDIM], ndim: u32, len: u32, storage_offset: u32) -> Self {
        Self {
            shape,
            stride,
            ndim,
            len,
            storage_offset,
            _padding: 0,
        }
    }
}
