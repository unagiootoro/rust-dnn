use std::ffi::c_void;

use crate::{
    basic::{cuda_fill_double, cuda_fill_float, cuda_fill_uint32_t},
    cuda::{
        CUDA_MEMCPY_DEVICE_TO_DEVICE, CUDA_MEMCPY_DEVICE_TO_HOST, CUDA_MEMCPY_HOST_TO_DEVICE,
        check_cuda_error, cudaDeviceSynchronize, cudaFree, cudaMalloc, cudaMemcpy,
    },
};

pub struct RawGpuBuffer {
    size: usize,
    ptr: *mut c_void,
    freed: bool,
}

impl RawGpuBuffer {
    pub unsafe fn new(size: usize) -> Self {
        let mut device_ptr: *mut c_void = std::ptr::null_mut();

        unsafe {
            // GPUメモリ確保
            let result = cudaMalloc(&mut device_ptr as *mut *mut c_void, size);
            check_cuda_error();
            assert_eq!(result, 0, "cudaMalloc failed");
        };

        Self {
            size,
            ptr: device_ptr,
            freed: false,
        }
    }

    pub fn from_vec(data: &Vec<u8>) -> Self {
        let mut raw_gpu_buffer = unsafe { RawGpuBuffer::new(data.len()) };
        raw_gpu_buffer.set_data(data);
        raw_gpu_buffer
    }

    pub fn copy(&mut self, buffer: &RawGpuBuffer) {
        if self.size != buffer.size {
            panic!("self.size = {}, buffer.size = {}", self.size, buffer.size);
        }

        let result = unsafe {
            cudaMemcpy(
                self.ptr,
                buffer.ptr as *const c_void,
                self.size,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
            )
        };
        check_cuda_error();
        assert_eq!(result, 0, "cudaMemcpy (D2D) failed");
    }

    pub fn set_data(&mut self, data: &[u8]) {
        if self.size != data.len() {
            panic!("self.len = {}, data.len() = {}", self.size, data.len());
        }

        // 転送: ホスト → デバイス
        let result = unsafe {
            cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                self.size,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        check_cuda_error();
        assert_eq!(result, 0, "cudaMemcpy (H2D) failed");
    }

    pub fn fill_u32(&mut self, value: u32, len: usize) {
        unsafe {
            cuda_fill_uint32_t(self.ptr as *mut u32, value, len as i32);
        }
        check_cuda_error();
    }

    pub fn fill_f32(&mut self, value: f32, len: usize) {
        unsafe {
            cuda_fill_float(self.ptr as *mut f32, value, len as i32);
        }
        check_cuda_error();
    }

    pub fn fill_f64(&mut self, value: f64, len: usize) {
        unsafe {
            cuda_fill_double(self.ptr as *mut f64, value, len as i32);
        }
        check_cuda_error();
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn to_vec(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.size);
        unsafe {
            cudaDeviceSynchronize();
            // 転送: デバイス → ホスト
            let result = cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                self.size,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            check_cuda_error();
            assert_eq!(result, 0, "cudaMemcpy (D2H) failed");
            data.set_len(self.size);
        }
        data
    }

    pub fn clone(&self) -> RawGpuBuffer {
        let mut gpu_buffer = unsafe { RawGpuBuffer::new(self.size) };
        gpu_buffer.copy(&self);
        gpu_buffer
    }

    pub fn free(&mut self) {
        if self.freed {
            return;
        }
        unsafe {
            cudaFree(self.ptr);
        }
        check_cuda_error();
        self.freed = true;
    }
}

impl Drop for RawGpuBuffer {
    fn drop(&mut self) {
        self.free();
    }
}
