use crate::{buffer_to_vec, create_wgpu_buffer_from_data, wgpu_dtype::{WgpuDType, WgpuDTypeKind}};

pub struct WgpuBuffer {
    pub raw: wgpu::Buffer,
    len: usize,
    dtype: WgpuDTypeKind,
}

impl WgpuBuffer {
    pub fn fill<T: WgpuDType>(len: usize, value: T) -> Self {
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(value);
        }
        Self::from_vec(vec)
    }

    pub fn from_vec<T: WgpuDType>(data: Vec<T>) -> Self {
        let len = data.len();
        let raw = create_wgpu_buffer_from_data(data);
        Self { raw, len, dtype: T::wgpu_dtype_kind() }
    }

    pub fn to_vec<T: WgpuDType>(&self) -> Vec<T> {
        assert_eq!(T::wgpu_dtype_kind(), self.dtype());
        buffer_to_vec(&self.raw, self.len)
    }

    pub fn dtype(&self) -> WgpuDTypeKind {
        self.dtype
    }
}
