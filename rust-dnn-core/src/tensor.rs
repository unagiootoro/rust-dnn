use std::{
    cell::RefCell,
    ops::{self, Deref},
    rc::Rc,
};

use crate::{
    cpu_storage::CpuStorage,
    dtype::DType,
    error::{Error, Result},
    layout::Layout,
    num::Num,
    storage::Storage,
};

pub struct TensorState {
    storage: Rc<RefCell<Storage>>,
    layout: Layout,
    dtype: DType,
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
    pub fn new(storage: Rc<RefCell<Storage>>, layout: Layout, dtype: DType) -> Self {
        let state = TensorState {
            storage,
            layout,
            dtype,
        };
        Self(Rc::new(state))
    }

    pub fn from_vec<T: Num>(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
        let len = Self::compute_len(&shape);
        if data.len() != len {
            let msg = format!(
                "Invalid data length(data len = {}, shape len = {})",
                data.len(),
                len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let cpu_storage = CpuStorage::from_vec(data);
        let storage = Rc::new(RefCell::new(Storage::CpuStorage(cpu_storage)));
        let stride = Self::compute_stride(&shape);
        let layout = Layout::new(shape, stride, 0);
        Ok(Self::new(storage, layout, T::dtype()))
    }

    pub fn arange(len: usize) -> Self {
        let mut data = Vec::new();
        for i in 0..len {
            data.push(i as f32);
        }
        let shape = vec![data.len()];
        let stride = Self::compute_stride(&shape);
        let cpu_storage = CpuStorage::from_vec(data);
        let storage = Rc::new(RefCell::new(Storage::CpuStorage(cpu_storage)));
        let layout = Layout::new(shape, stride, 0);
        Self::new(storage, layout, DType::F32)
    }

    fn compute_len(shape: &[usize]) -> usize {
        let mut len = 1;
        for dim in shape {
            len *= dim;
        }
        len
    }

    fn compute_stride(shape: &[usize]) -> Vec<usize> {
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

    pub fn to_vec<T: Num>(&self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len());
        let Storage::CpuStorage(input_cpu_storage) = &*self.storage.borrow();
        let input_data = input_cpu_storage.as_slice();

        for i in 0..self.len() {
            let offset = Self::compute_offset(&self.layout, i);
            vec.push(input_data[offset]);
        }
        vec
    }

    pub fn shape(&self) -> &[usize] {
        &self.layout.shape()
    }

    pub fn stride(&self) -> &[usize] {
        &self.layout.stride()
    }

    pub fn len(&self) -> usize {
        self.layout.len()
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    pub fn storage_offset(&self) -> usize {
        self.layout.storage_offset()
    }

    fn add_impl(&self, rhs: &Tensor) -> Result<Self> {
        let broadcasted_shape = Self::broadcast_shape(self.shape(), rhs.shape())?;
        let lhs = self.broadcast_to(broadcasted_shape.clone())?;
        let rhs = rhs.broadcast_to(broadcasted_shape.clone())?;
        let lhs_storage = &*lhs.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();

        let output_storage = match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                Self::op_add::<f32>(lhs_storage, rhs_storage, &lhs.layout, &rhs.layout)
            }
            (DType::U32, DType::U32) => {
                Self::op_add::<u32>(lhs_storage, rhs_storage, &lhs.layout, &rhs.layout)
            }
            _ => Err(Error::ArgumentsError {
                msg: "Invalid dtype".to_string(),
            }),
        }?;

        let stride = Self::compute_stride(&broadcasted_shape);
        let layout = Layout::new(broadcasted_shape, stride, 0);
        let output = Tensor::new(Rc::new(RefCell::new(output_storage)), layout, self.dtype);
        Result::Ok(output)
    }

    fn op_add<T: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage> {
        Self::map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a + b
        })
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let new_shape_len = Self::compute_len(&shape);
        if self.len() != new_shape_len {
            let msg = format!(
                "Length mismatch (from shape = {:?}, from shape len = {}, to shape = {:?}, to shape len = {})",
                self.shape(),
                self.len(),
                shape,
                new_shape_len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let input = self.contiguous()?;
        let stride = Self::compute_stride(&shape);
        let layout = Layout::new(shape, stride, input.storage_offset());
        let output = Tensor::new(input.storage.clone(), layout, self.dtype);
        Result::Ok(output)
    }

    pub fn permuted_axes(&self, axes: &[usize]) -> Result<Self> {
        if self.ndim() != axes.len() {
            let msg = format!(
                "Mismatch dims(self.ndim() = {}, axes.len() = {})",
                self.ndim(),
                axes.len()
            );
            return Err(Error::ArgumentsError { msg });
        }
        let mut new_shape = Vec::new();
        let mut new_stride = Vec::new();
        for axis in axes {
            new_shape.push(self.shape()[*axis]);
            new_stride.push(self.stride()[*axis]);
        }
        let layout = Layout::new(new_shape, new_stride, self.storage_offset());
        let output = Tensor::new(self.storage.clone(), layout, self.dtype);
        Result::Ok(output)
    }

    pub fn reversed_axes(&self) -> Result<Self> {
        let mut axes = Vec::new();
        for axis in (0..self.ndim()).rev() {
            axes.push(axis);
        }
        self.permuted_axes(&axes)
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Result<Self> {
        let mut input_shape = Vec::new();
        if self.ndim() < shape.len() {
            for _ in 0..(shape.len() - self.ndim()) {
                input_shape.push(1);
            }
        }
        for dim in self.shape() {
            input_shape.push(*dim);
        }
        let input = self.reshape(input_shape.clone())?;
        let stride = Self::compute_broadcast_stride(input.shape(), input.stride(), &shape)?;
        let layout = Layout::new(shape, stride, input.storage_offset());
        let output = Tensor::new(Rc::clone(&input.storage), layout, self.dtype);
        Ok(output)
    }

    fn compute_broadcast_stride(
        original_shape: &[usize],
        original_strides: &[usize],
        target_shape: &[usize],
    ) -> Result<Vec<usize>> {
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
        for (&dim, &stride) in self.shape().iter().rev().zip(self.stride().iter().rev()) {
            if stride != expected_stride {
                return false;
            }
            expected_stride *= dim;
        }
        true
    }

    pub fn contiguous(&self) -> Result<Self> {
        match self.dtype {
            DType::F32 => self.contiguous_impl::<f32>(),
            DType::U32 => self.contiguous_impl::<u32>(),
        }
    }

    fn contiguous_impl<T: Num>(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let mut output_data = Vec::with_capacity(self.len());
        let Storage::CpuStorage(input_cpu_storage) = &*self.storage.borrow();
        let input_data = input_cpu_storage.as_slice::<T>();

        for i in 0..self.len() {
            let offset = Self::compute_offset(&self.layout, i);
            output_data.push(input_data[offset]);
        }

        let output_storage = Rc::new(RefCell::new(Storage::CpuStorage(CpuStorage::from_vec(
            output_data,
        ))));
        let layout = Layout::new(
            self.shape().to_vec(),
            Self::compute_stride(&self.shape()),
            0,
        );
        let output = Tensor::new(output_storage, layout, self.dtype);
        Ok(output)
    }

    fn compute_offset(layout: &Layout, mut linear_index: usize) -> usize {
        let mut offset = 0;
        for i in (0..layout.ndim()).rev() {
            if layout.stride()[i] > 0 {
                let idx = linear_index % layout.shape()[i];
                offset += idx * layout.stride()[i];
            }
            linear_index /= layout.shape()[i];
        }
        layout.storage_offset() + offset
    }

    fn map_arg2<T: Num, F>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        f: F,
    ) -> Result<Storage>
    where
        F: Fn(T, T) -> T + Sync,
    {
        let Storage::CpuStorage(a_cpu_storage) = lhs_storage;
        let Storage::CpuStorage(b_cpu_storage) = rhs_storage;

        let a_data = a_cpu_storage.as_slice::<T>();
        let b_data = b_cpu_storage.as_slice::<T>();

        let output_data: Vec<T> = (0..lhs_layout.len())
            .into_iter()
            .map(|i| {
                let a_index = Self::compute_offset(&lhs_layout, i);
                let b_index = Self::compute_offset(&rhs_layout, i);
                f(a_data[a_index], b_data[b_index])
            })
            .collect();
        let output_storage = Storage::CpuStorage(CpuStorage::from_vec(output_data));
        Result::Ok(output_storage)
    }

    fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
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
            .flat_map(|a| a.to_vec::<f32>())
            .collect::<Vec<_>>();
        tensor::Tensor::from_vec(data, shape).unwrap()
    }};
    ( $($x:expr),+ $(,)? ) => {{
        let data = vec![$($x),+];
        let shape = vec![data.len()];
        tensor::Tensor::from_vec(data, shape).unwrap()
    }};
}

impl ops::Add<Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn add(self, rhs: Tensor) -> Result<Tensor> {
        self.add_impl(&rhs)
    }
}

impl ops::Add<Tensor> for Result<Tensor> {
    type Output = Result<Tensor>;

    fn add(self, rhs: Tensor) -> Result<Tensor> {
        self?.add_impl(&rhs)
    }
}
