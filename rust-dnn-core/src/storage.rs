use crate::cpu_storage::CpuStorage;

pub enum Storage {
    CpuStorage(CpuStorage),
}

impl Storage {
    pub fn to_vec(&self) -> Vec<f32> {
        match self {
            Storage::CpuStorage(storage) => storage.as_slice::<f32>().to_vec(),
        }
    }
}
