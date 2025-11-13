pub const MAX_NDIM: usize = 8;

# [repr(C)]
pub struct CLayout {
    pub shape: [usize; MAX_NDIM],
    pub stride: [usize; MAX_NDIM],
    pub ndim: usize,
    pub len: usize,
    pub storage_offset: usize,
}

# [repr(C)]
pub struct NDimArray {
    pub data: [usize; MAX_NDIM],
    pub ndim: usize,
}
