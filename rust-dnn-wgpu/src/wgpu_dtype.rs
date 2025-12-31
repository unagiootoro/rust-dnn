use bytemuck::{AnyBitPattern, NoUninit};

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum WgpuDTypeKind {
    U32,
    F32,
}

pub trait WgpuDType : NoUninit + AnyBitPattern {
    fn wgpu_dtype_kind() -> WgpuDTypeKind;
}

impl WgpuDType for u32 {
    fn wgpu_dtype_kind() -> WgpuDTypeKind {
        WgpuDTypeKind::U32
    }
}

impl WgpuDType for f32 {
    fn wgpu_dtype_kind() -> WgpuDTypeKind {
        WgpuDTypeKind::F32
    }
}
