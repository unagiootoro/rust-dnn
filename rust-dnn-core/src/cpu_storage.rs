pub struct CpuStorage {
    data: Vec<f32>,
}

impl CpuStorage {
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }
}
