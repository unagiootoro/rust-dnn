pub struct CpuStorage {
    data: Vec<u8>,
}

impl CpuStorage {
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }

    pub fn from_vec<T>(vec: Vec<T>) -> Self {
        Self::new(Self::cast_vec_t_to_vec_u8(vec))
    }

    pub fn to_vec(&self) -> Vec<u8> {
        self.data.clone()
    }

    pub fn data(&self) -> &Vec<u8> {
        &self.data
    }

    pub fn as_slice<T>(&self) -> &[T] {
        Self::cast_slice(&self.data)
    }

    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        Self::cast_slice_mut(&mut self.data)
    }

    fn cast_slice<T1, T2>(data: &[T1]) -> &[T2] {
        use std::mem::size_of;
        use std::slice;
        let ptr = data.as_ptr() as *const T2;
        let len = data.len() * size_of::<T1>() / size_of::<T2>();
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    fn cast_slice_mut<T1, T2>(data: &mut [T1]) -> &mut [T2] {
        use std::mem::size_of;
        use std::slice;
        let ptr = data.as_mut_ptr() as *mut T2;
        let len = data.len() * size_of::<T1>() / size_of::<T2>();
        unsafe { slice::from_raw_parts_mut(ptr, len) }
    }

    fn cast_vec_t_to_vec_u8<T>(mut v: Vec<T>) -> Vec<u8> {
        // メモリサイズ計算
        let elem_size = std::mem::size_of::<T>();
        let len = v.len() * elem_size;
        let capacity = v.capacity() * elem_size;

        let ptr = v.as_mut_ptr() as *mut u8;

        // Vec<T> の解放を防ぐために忘れる
        std::mem::forget(v);

        // 新しい Vec<u8> を再構築
        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    }
}
