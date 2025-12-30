use std::marker::PhantomData;

use crate::{buffer_to_vec, create_wgpu_buffer_from_data};

pub struct WgpuBuffer {
    pub raw: wgpu::Buffer,
    len: usize,
}

impl WgpuBuffer {
    pub fn zeros_u32(len: usize) -> Self {
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(0);
        }
        Self::from_vec_u32(vec)
    }

    pub fn zeros_f32(len: usize) -> Self {
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(0.0);
        }
        Self::from_vec_f32(vec)
    }

    pub fn from_vec_u32(data: Vec<u32>) -> Self {
        let len = data.len();
        let raw = create_wgpu_buffer_from_data(data);
        Self { raw, len }
    }

    pub fn from_vec_f32(data: Vec<f32>) -> Self {
        let len = data.len();
        let raw = create_wgpu_buffer_from_data(data);
        Self { raw, len }
    }

    pub fn to_vec_u32(&self) -> Vec<u32> {
        buffer_to_vec(&self.raw, self.len)
    }

    pub fn to_vec_f32(&self) -> Vec<f32> {
        buffer_to_vec(&self.raw, self.len)
    }
}
