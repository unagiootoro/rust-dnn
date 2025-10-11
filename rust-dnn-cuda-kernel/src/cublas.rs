use libc::{c_int, c_void};
use std::cell::RefCell;

thread_local! {
    pub static CUBLAS_HANDLE: RefCell<Option<*mut c_void>> = RefCell::new(None);
}

// cublasOperation_t
pub const CUBLAS_OP_N: i32 = 0;
pub const CUBLAS_OP_T: i32 = 1;

unsafe extern "C" {
    pub fn cublasCreate_v2(handle: *mut *mut c_void) -> i32;
    pub fn cublasDestroy_v2(handle: *mut c_void) -> i32;
    pub fn cublasSgemm_v2(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        B: *const f32,
        ldb: c_int,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
    ) -> i32;

    pub fn cublasSgemmStridedBatched(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        A: *const f32,
        lda: c_int,
        strideA: i64,
        B: *const f32,
        ldb: c_int,
        strideB: i64,
        beta: *const f32,
        C: *mut f32,
        ldc: c_int,
        strideC: i64,
        batchCount: i32,
    ) -> i32;
}
