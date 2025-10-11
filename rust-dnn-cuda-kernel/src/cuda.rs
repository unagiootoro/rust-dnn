use libc::{c_char, c_void, size_t};
use std::ffi::CStr;

pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

unsafe extern "C" {
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: size_t) -> i32;
    pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: size_t, kind: i32) -> i32;
    pub fn cudaFree(ptr: *mut c_void) -> i32;
    pub fn cudaDeviceSynchronize() -> i32;
    pub fn cudaGetLastError() -> CudaError;
    pub fn cudaGetErrorString(error: CudaError) -> *const c_char;
}

// CUDAのエラーコード
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CudaError {
    Success = 0,
    MissingConfiguration = 1,
    MemoryAllocation = 2,
    InitializationError = 3,
    // 必要に応じて追加
    Unknown = 30,
    InvalidValue = 11,
    UnknownError = 999,
}

// エラーチェック関数
pub fn check_cuda_error() {
    unsafe {
        let err = cudaGetLastError();
        if err != CudaError::Success {
            let err_str = cudaGetErrorString(err);
            let c_str = CStr::from_ptr(err_str);
            eprintln!("CUDA Error: {:?}", c_str);
            panic!();
        }
    }
}
