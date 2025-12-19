use std::{cell::RefCell, ffi::c_void, marker::PhantomData, rc::Rc};

use crate::{gpu_memory_allocator::GpuMemoryAllocator, gpu_memory_block::GpuMemoryBlock};

thread_local! {
    pub static GPU_MEMORY_ALLOCATOR: RefCell<GpuMemoryAllocator> = RefCell::new(GpuMemoryAllocator::new());
}

pub struct GPUBuffer<T> {
    memory_block: Rc<RefCell<GpuMemoryBlock>>,
    len: usize,
    marker: PhantomData<T>,
}

impl<T> GPUBuffer<T> {
    pub unsafe fn new(len: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        GPU_MEMORY_ALLOCATOR.with(|allocator| {
            let memory_block = unsafe { allocator.borrow_mut().allocate(size) };
            Self {
                memory_block,
                len,
                marker: PhantomData,
            }
        })
    }

    pub fn from_fill_u32(len: usize, value: u32) -> Self {
        let mut gpu_buffer = unsafe { GPUBuffer::new(len) };
        gpu_buffer.fill_u32(value);
        gpu_buffer
    }

    pub fn from_fill_f32(len: usize, value: f32) -> Self {
        let mut gpu_buffer = unsafe { GPUBuffer::new(len) };
        gpu_buffer.fill_f32(value);
        gpu_buffer
    }

    pub fn from_fill_f64(len: usize, value: f64) -> Self {
        let mut gpu_buffer = unsafe { GPUBuffer::new(len) };
        gpu_buffer.fill_f64(value);
        gpu_buffer
    }

    pub fn from_vec(data: &Vec<T>) -> Self {
        let mut gpu_buffer = unsafe { GPUBuffer::new(data.len()) };
        gpu_buffer.set_data(data);
        gpu_buffer
    }

    pub fn from_slice(slice: &[T]) -> Self {
        let mut gpu_buffer = unsafe { GPUBuffer::new(slice.len()) };
        gpu_buffer.set_data(slice);
        gpu_buffer
    }

    pub fn copy(&mut self, buffer: &GPUBuffer<T>) {
        if self.len != buffer.len {
            panic!("self.len = {}, buffer.len = {}", self.len, buffer.len);
        }
        self.memory_block
            .borrow_mut()
            .buffer
            .copy(&buffer.memory_block.borrow().buffer);
    }

    pub fn set_data(&mut self, data: &[T]) {
        if self.len != data.len() {
            panic!("self.len = {}, data.len() = {}", self.len, data.len());
        }
        self.memory_block
            .borrow_mut()
            .buffer
            .set_data(Self::cast_slice(&data));
    }

    pub fn fill_u32(&mut self, value: u32) {
        self.memory_block
            .borrow_mut()
            .buffer
            .fill_u32(value, self.len);
    }

    pub fn fill_f32(&mut self, value: f32) {
        self.memory_block
            .borrow_mut()
            .buffer
            .fill_f32(value, self.len);
    }

    pub fn fill_f64(&mut self, value: f64) {
        self.memory_block
            .borrow_mut()
            .buffer
            .fill_f64(value, self.len);
    }

    pub fn ptr(&self) -> *mut c_void {
        self.memory_block.borrow_mut().buffer.ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn to_vec(&self) -> Vec<T> {
        Self::cast_vec_u8_to_vec_t(self.memory_block.borrow().buffer.to_vec())
    }

    fn cast_slice<T1, T2>(data: &[T1]) -> &[T2] {
        use std::mem::size_of;
        use std::slice;
        let ptr = data.as_ptr() as *const T2;
        let len = data.len() * size_of::<T1>() / size_of::<T2>();
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    fn cast_vec_u8_to_vec_t(v: Vec<u8>) -> Vec<T> {
        use std::mem::{ManuallyDrop, size_of};

        assert!(
            v.len() % size_of::<T>() == 0,
            "長さが {}byte 単位で割り切れません",
            size_of::<T>()
        );

        let mut v = ManuallyDrop::new(v);
        let ptr = v.as_mut_ptr() as *mut T;
        let len = v.len() / size_of::<T>();
        let cap = v.capacity() / size_of::<T>();

        unsafe { Vec::from_raw_parts(ptr, len, cap) }
    }

    pub fn clone(&self) -> GPUBuffer<T> {
        let mut gpu_buffer = unsafe { GPUBuffer::new(self.len) };
        gpu_buffer.copy(&self);
        gpu_buffer
    }
}

impl<T> Drop for GPUBuffer<T> {
    fn drop(&mut self) {
        GPU_MEMORY_ALLOCATOR.with(|allocator| {
            allocator.borrow_mut().free(&self.memory_block);
        })
    }
}
