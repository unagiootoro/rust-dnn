pub struct Layout {
    shape: Vec<usize>,
    len: usize,
    stride: Vec<usize>,
    storage_offset: usize,
}

impl Layout {
    pub fn new(shape: Vec<usize>, stride: Vec<usize>, storage_offset: usize) -> Self {
        let len = Self::compute_len(&shape);
        Self {
            shape,
            stride,
            len,
            storage_offset,
        }
    }

    fn compute_len(shape: &[usize]) -> usize {
        let mut len = 1;
        for dim in shape {
            len *= dim;
        }
        len
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }

    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}
