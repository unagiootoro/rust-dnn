use libc::{c_char, c_void, size_t};
use std::{cell::RefCell, ffi::CStr, ptr::null_mut};

thread_local! {
    pub static CUDNN_HANDLE: RefCell<Option<*mut c_void>> = RefCell::new(None);
}

#[link(name = "cudnn")]
unsafe extern "C" {
    pub fn cudnnCreate(handle: *mut *mut c_void) -> i32;
    pub fn cudnnDestroy(handle: *mut c_void) -> i32;

    pub fn cudnnCreateTensorDescriptor(desc: *mut *mut c_void) -> i32;
    pub fn cudnnSetTensor4dDescriptor(
        desc: *mut c_void,
        format: i32,
        dataType: i32,
        n: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> i32;

    pub fn cudnnCreateFilterDescriptor(desc: *mut *mut c_void) -> i32;
    pub fn cudnnSetFilter4dDescriptor(
        desc: *mut c_void,
        dataType: i32,
        format: i32,
        k: i32,
        c: i32,
        h: i32,
        w: i32,
    ) -> i32;

    pub fn cudnnCreateConvolutionDescriptor(desc: *mut *mut c_void) -> i32;
    pub fn cudnnSetConvolution2dDescriptor(
        convDesc: *mut c_void,
        pad_h: i32,
        pad_w: i32,
        u: i32,
        v: i32,
        dilation_h: i32,
        dilation_w: i32,
        mode: i32,
        computeType: i32,
    ) -> i32;

    pub fn cudnnGetConvolutionForwardWorkspaceSize(
        handle: *mut c_void,
        xDesc: *mut c_void,
        wDesc: *mut c_void,
        convDesc: *mut c_void,
        yDesc: *mut c_void,
        algo: i32,
        sizeInBytes: *mut size_t,
    ) -> i32;

    pub fn cudnnConvolutionForward(
        handle: *mut c_void,
        alpha: *const c_void,
        xDesc: *mut c_void,
        x: *const c_void,
        wDesc: *mut c_void,
        w: *const c_void,
        convDesc: *mut c_void,
        algo: i32,
        workspace: *mut c_void,
        workspaceSizeInBytes: size_t,
        beta: *const c_void,
        yDesc: *mut c_void,
        y: *mut c_void,
    ) -> i32;

    pub fn cudnnConvolutionBackwardData(
        handle: *mut c_void,
        alpha: *const c_void,
        wDesc: *mut c_void,
        w: *const c_void,
        dyDesc: *mut c_void,
        dy: *const c_void,
        convDesc: *mut c_void,
        algo: i32,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: size_t,
        beta: *const c_void,
        dxDesc: *mut c_void,
        dx: *mut c_void,
    ) -> i32;

    pub fn cudnnConvolutionBackwardFilter(
        handle: *mut c_void,
        alpha: *const c_void,
        xDesc: *mut c_void,
        x: *const c_void,
        dyDesc: *mut c_void,
        dy: *const c_void,
        convDesc: *mut c_void,
        algo: i32,
        workSpace: *mut c_void,
        workSpaceSizeInBytes: size_t,
        beta: *const c_void,
        dwDesc: *mut c_void,
        dw: *mut c_void,
    ) -> i32;

    pub fn cudnnGetConvolution2dForwardOutputDim(
        convDesc: *mut c_void,
        inputTensorDesc: *mut c_void,
        filterDesc: *mut c_void,
        n: *mut i32,
        c: *mut i32,
        h: *mut i32,
        w: *mut i32,
    ) -> i32;

    pub fn cudnnGetErrorString(status: i32) -> *const c_char;
}

pub const CUDNN_TENSOR_NCHW: i32 = 0;
pub const CUDNN_DATA_FLOAT: i32 = 0;
pub const CUDNN_CROSS_CORRELATION: i32 = 1;
pub const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM: i32 = 0;

pub fn check_cudnn_error(status: i32) {
    if status != 0 {
        unsafe {
            let err_str = cudnnGetErrorString(status);
            let c_str = CStr::from_ptr(err_str);
            eprintln!("CUDNN Error: {:?}", c_str);
            panic!();
        }
    }
}
