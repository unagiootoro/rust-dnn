use std::{cell::RefCell, ops::Deref, rc::Rc};

use crate::{cpu_storage::CpuStorage, error::Error, storage::Storage};

pub struct TensorState {
    storage: Rc<RefCell<Storage>>,
    shape: Vec<usize>,
    len: usize,
    stride: Vec<usize>,
    storage_offset: usize,
}

#[derive(Clone)]
pub struct Tensor(Rc<TensorState>);

impl Deref for Tensor {
    type Target = TensorState;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl Tensor {
    pub fn new(
        storage: Rc<RefCell<Storage>>,
        shape: Vec<usize>,
        stride: Vec<usize>,
        storage_offset: usize,
    ) -> Self {
        let len = Self::compute_len(&shape);
        let state = TensorState {
            storage,
            shape,
            len,
            stride,
            storage_offset,
        };
        Self(Rc::new(state))
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, Error> {
        let len = Self::compute_len(&shape);
        if data.len() != len {
            let msg = format!(
                "Invalid data length(data len = {}, shape len = {})",
                data.len(),
                len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let cpu_storage = CpuStorage::new(data);
        let storage = Rc::new(RefCell::new(Storage::CpuStorage(cpu_storage)));
        let stride = Self::compute_stride(&shape);
        Ok(Self::new(storage, shape, stride, 0))
    }

    fn compute_len(shape: &[usize]) -> usize {
        let mut len = 1;
        for dim in shape {
            len *= dim;
        }
        len
    }

    fn compute_stride(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides = Vec::new();
        for i in 0..shape.len() {
            let mut stride = 1;
            for j in (i + 1)..shape.len() {
                stride *= shape[j];
            }
            strides.push(stride);
        }
        strides
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.storage.borrow().to_vec()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn add(&self, other: &Tensor) -> Result<Self, Error> {
        self.map_arg2(other, |a, b| a + b)
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self, Error> {
        let new_shape_len = Self::compute_len(&shape);
        if self.len != new_shape_len {
            let msg = format!(
                "Length mismatch (from shape = {:?}, from shape len = {}, to shape = {:?}, to shape len = {})",
                self.shape, self.len, shape, new_shape_len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let input = self.contiguous()?;
        let stride = Self::compute_stride(&shape);
        let output = Tensor::new(input.storage.clone(), shape, stride, input.storage_offset);
        Result::Ok(output)
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Self, Error> {
        let mut shape2 = Vec::new();
        if self.shape.len() < shape.len() {
            for _ in 0..(self.shape.len() - shape.len()) {
                shape2.push(1);
            }
        }
        for dim in &self.shape {
            shape2.push(*dim);
        }
        let input = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        let input = input.reshape(shape2.clone())?;

        let stride = Self::compute_broadcast_stride(&input.shape, &input.stride, &shape2)?;
        let output = Tensor::new(
            Rc::clone(&self.storage),
            shape2,
            stride,
            self.storage_offset,
        );
        Ok(output)
    }

    fn compute_broadcast_stride(
        original_shape: &Vec<usize>,
        original_strides: &Vec<usize>,
        target_shape: &Vec<usize>,
    ) -> Result<Vec<usize>, Error> {
        if target_shape.len() < original_shape.len() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid broadcast shape({:?})", target_shape),
            });
        }

        let mut result = vec![0; target_shape.len()];
        let dim_offset = target_shape.len() - original_shape.len();

        for i in 0..target_shape.len() {
            let target_dim = target_shape[i];
            let (orig_dim, orig_stride) = if i < dim_offset {
                (1, 0)
            } else {
                (
                    original_shape[i - dim_offset],
                    original_strides[i - dim_offset],
                )
            };

            if orig_dim == target_dim {
                result[i] = orig_stride;
            } else if orig_dim == 1 {
                result[i] = 0;
            } else {
                return Err(Error::ArgumentsError {
                    msg: format!("Invalid broadcast shape({:?})", target_shape),
                });
            }
        }

        Ok(result)
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for (&dim, &stride) in self.shape.iter().rev().zip(self.stride.iter().rev()) {
            if stride != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }

    pub fn contiguous(&self) -> Result<Self, Error> {
        let mut output_data = Vec::with_capacity(self.len);
        let Storage::CpuStorage(input_cpu_storage) = &*self.storage.borrow();
        let input_data = input_cpu_storage.data();

        for i in 0..self.len {
            let offset = Self::compute_offset(self.storage_offset, &self.shape, &self.stride, i);
            output_data.push(input_data[offset]);
        }

        let output_storage = Rc::new(RefCell::new(Storage::CpuStorage(CpuStorage::new(
            output_data,
        ))));
        let output = Tensor::new(
            output_storage,
            self.shape.clone(),
            Self::compute_stride(&self.shape),
            0,
        );
        Ok(output)
    }

    fn compute_offset(
        base_offset: usize,
        shape: &[usize],
        stride: &[usize],
        mut linear_index: usize,
    ) -> usize {
        let mut offset = 0;
        for i in (0..shape.len()).rev() {
            if stride[i] > 0 {
                let idx = linear_index % shape[i];
                offset += idx * stride[i];
            }
            linear_index /= shape[i];
        }
        base_offset + offset
    }

    fn map_arg2<F>(&self, other: &Tensor, f: F) -> Result<Tensor, Error>
    where
        F: Fn(f32, f32) -> f32 + Sync,
    {
        let broadcasted_shape = Self::broadcast_shape(self.shape(), other.shape())?;
        let a = self.broadcast_to(broadcasted_shape.clone())?;
        let b = other.broadcast_to(broadcasted_shape.clone())?;
        let Storage::CpuStorage(a_cpu_storage) = &*self.storage.borrow();
        let Storage::CpuStorage(b_cpu_storage) = &*other.storage.borrow();

        let a_data = a_cpu_storage.data();
        let b_data = b_cpu_storage.data();

        let len = Self::compute_len(&broadcasted_shape);
        let output_data: Vec<f32> = (0..len)
            .into_iter()
            .map(|i| {
                let a_index = Self::compute_offset(a.storage_offset, &a.shape, &a.stride, i);
                let b_index = Self::compute_offset(b.storage_offset, &b.shape, &b.stride, i);
                f(a_data[a_index], b_data[b_index])
            })
            .collect();
        let output_storage = Rc::new(RefCell::new(Storage::CpuStorage(CpuStorage::new(
            output_data,
        ))));
        let output = Tensor::new(output_storage, self.shape.clone(), self.stride.clone(), 0);
        Result::Ok(output)
    }

    fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, Error> {
        let mut c = Vec::new();
        let ndim = if a.len() < b.len() { b.len() } else { a.len() };
        for i in 0..ndim {
            let a_dim = if ndim - i > a.len() {
                1
            } else {
                a[i - (ndim - a.len())]
            };
            let b_dim = if ndim - i > b.len() {
                1
            } else {
                b[i - (ndim - b.len())]
            };
            if a_dim == 1 && b_dim > 1 {
                c.push(b_dim);
            } else if a_dim > 1 && b_dim == 1 {
                c.push(a_dim);
            } else if a_dim == 1 && b_dim == 1 {
                c.push(1);
            } else if a_dim == b_dim {
                c.push(a_dim);
            } else {
                let msg = format!("broadcast failed(a = {:?}, b = {:?})", a, b);
                return Err(Error::ArgumentsError { msg });
            }
        }
        Ok(c)
    }
}

#[macro_export]
macro_rules! tensor {
    ( $( [ $($inner:tt)* ] ),+ $(,)? ) => {{
        let arrays = vec![$( tensor![ $($inner)* ] ),+];
        let shape = {
            let mut s = arrays[0].shape().to_vec();
            s.insert(0, arrays.len());
            s
        };
        let data = arrays
            .into_iter()
            .flat_map(|a| a.to_vec())
            .collect::<Vec<_>>();
        tensor::Tensor::from_vec(data, shape).unwrap()
    }};
    ( $($x:expr),+ $(,)? ) => {{
        let data = vec![$($x),+];
        let shape = vec![data.len()];
        tensor::Tensor::from_vec(data, shape).unwrap()
    }};
}
