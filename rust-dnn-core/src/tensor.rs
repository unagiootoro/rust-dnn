use std::{
    cell::RefCell,
    collections::HashSet,
    marker::PhantomData,
    ops::{self, Deref, Range},
    rc::Rc,
};

use crate::{
    cpu_storage::CpuStorage,
    dtype::DType,
    error::{Error, Result},
    float::Float,
    gradients::Gradients,
    layout::Layout,
    num::Num,
    op::Op,
    storage::Storage,
};

thread_local! {
    pub static TENSOR_ID_COUNTER: RefCell<usize> = RefCell::new(0);
}

pub struct TensorState<T: Num> {
    id: usize,
    storage: Rc<RefCell<Storage>>,
    layout: Layout,
    dtype: DType,
    is_requires_grad: bool,
    op: Option<Op<T>>,
    _marker: PhantomData<T>,
}

#[derive(Clone)]
pub struct Tensor<T: Num>(Rc<TensorState<T>>);

impl<T: Num> Deref for Tensor<T> {
    type Target = TensorState<T>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<T: Num> Tensor<T> {
    pub fn new(
        storage: Rc<RefCell<Storage>>,
        layout: Layout,
        dtype: DType,
        requires_grad: bool,
        op: Option<Op<T>>,
    ) -> Self {
        let id = TENSOR_ID_COUNTER.with(|counter| *counter.borrow());
        TENSOR_ID_COUNTER.with(|counter| *counter.borrow_mut() = id + 1);
        let state = TensorState::<T> {
            id,
            storage,
            layout,
            dtype,
            is_requires_grad: requires_grad,
            op,
            _marker: PhantomData,
        };
        Self(Rc::new(state))
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self> {
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
        Ok(Self::new(storage, layout, T::dtype(), false, None))
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::fill(shape, T::zero())
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        Self::fill(shape, T::one())
    }

    pub fn fill(shape: Vec<usize>, value: T) -> Self {
        let mut data = Vec::new();
        for _ in 0..Self::compute_len(&shape) {
            data.push(value);
        }
        let stride = Self::compute_stride(&shape);
        let cpu_storage = CpuStorage::from_vec(data);
        let storage = Rc::new(RefCell::new(Storage::CpuStorage(cpu_storage)));
        let layout = Layout::new(shape, stride, 0);
        Self::new(storage, layout, T::dtype(), false, None)
    }

    pub fn arange(range: Range<isize>) -> Self {
        let mut data = Vec::new();
        for i in range {
            data.push(T::from_isize(i));
        }
        let shape = vec![data.len()];
        let stride = Self::compute_stride(&shape);
        let cpu_storage = CpuStorage::from_vec(data);
        let storage = Rc::new(RefCell::new(Storage::CpuStorage(cpu_storage)));
        let layout = Layout::new(shape, stride, 0);
        Self::new(storage, layout, T::dtype(), false, None)
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

    pub fn to_vec(&self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.len());
        let Storage::CpuStorage(input_cpu_storage) = &*self.storage.borrow();
        let input_data = input_cpu_storage.as_slice();

        for i in 0..self.len() {
            let offset = Self::compute_offset(&self.layout, i);
            vec.push(input_data[offset]);
        }
        vec
    }

    pub fn id(&self) -> usize {
        self.id
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

    pub fn is_requires_grad(&self) -> bool {
        self.is_requires_grad
    }

    pub fn requires_grad(self) -> Self {
        Self::new(
            self.storage.clone(),
            self.layout.clone(),
            self.dtype,
            true,
            self.op.clone(),
        )
    }

    pub fn detach(self) -> Self {
        if !self.is_requires_grad() && self.op.is_none() {
            self
        } else {
            Self::new(
                self.storage.clone(),
                self.layout.clone(),
                self.dtype,
                false,
                None,
            )
        }
    }

    fn add_impl(&self, rhs: &Tensor<T>) -> Result<Self> {
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
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.dtype,
            self.is_requires_grad || rhs.is_requires_grad,
            None,
        );
        Result::Ok(output)
    }

    fn op_add<T2: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage> {
        Self::map_arg2::<T2, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
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
        let op = if self.is_requires_grad {
            Some(Op::Reshape(self.clone()))
        } else {
            None
        };
        let output = Tensor::new(
            input.storage.clone(),
            layout,
            self.dtype,
            self.is_requires_grad,
            op,
        );
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
        let output = Tensor::new(
            self.storage.clone(),
            layout,
            self.dtype,
            self.is_requires_grad,
            None,
        );
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
        let output = Tensor::new(
            Rc::clone(&input.storage),
            layout,
            self.dtype,
            self.is_requires_grad,
            None,
        );
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

    fn contiguous_impl<T2: Num>(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let mut output_data = Vec::with_capacity(self.len());
        let Storage::CpuStorage(input_cpu_storage) = &*self.storage.borrow();
        let input_data = input_cpu_storage.as_slice::<T2>();

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
        let output = Tensor::new(
            output_storage,
            layout,
            self.dtype,
            self.is_requires_grad,
            None,
        );
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

    fn map_arg2<T2: Num, F>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        f: F,
    ) -> Result<Storage>
    where
        F: Fn(T2, T2) -> T2 + Sync,
    {
        let Storage::CpuStorage(a_cpu_storage) = lhs_storage;
        let Storage::CpuStorage(b_cpu_storage) = rhs_storage;

        let a_data = a_cpu_storage.as_slice::<T2>();
        let b_data = b_cpu_storage.as_slice::<T2>();

        let output_data: Vec<T2> = (0..lhs_layout.len())
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

impl<T: Float> Tensor<T> {
    pub fn sorted_nodes(&self) -> Vec<Tensor<T>> {
        let mut nodes: Vec<(usize, Tensor<T>)> = Vec::new();

        let mut seen_set = HashSet::new();
        let mut work_nodes = Vec::<(usize, Tensor<T>)>::new();
        Self::add_work_node(0, self.clone(), &mut work_nodes, &mut seen_set);

        while work_nodes.len() > 0 {
            let (depth, op_tensor) = work_nodes.pop().unwrap();
            if !op_tensor.is_requires_grad {
                continue;
            }
            let Some(op) = op_tensor.op.clone() else {
                continue;
            };

            nodes.push((depth, op_tensor));

            match op {
                Op::Reshape(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
            }
        }

        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        nodes.into_iter().map(|a| a.1).collect()
    }

    fn add_work_node(
        depth: usize,
        node: Tensor<T>,
        work_nodes: &mut Vec<(usize, Tensor<T>)>,
        seen_set: &mut HashSet<usize>,
    ) {
        if !seen_set.contains(&node.id()) {
            seen_set.insert(node.id());
            work_nodes.push((depth, node.clone()));
        }
    }

    pub fn backward(&self) -> Result<Gradients<T>> {
        let mut grads = Gradients::new();
        if !self.is_requires_grad {
            return Ok(grads);
        }

        let gy = Tensor::ones(self.shape().to_vec());
        grads.insert(self, gy.clone());

        let nodes = self.sorted_nodes();

        for node in nodes {
            if !node.is_requires_grad {
                continue;
            }
            let Some(op) = node.op.clone() else {
                continue;
            };

            let gy = grads.get(&node).unwrap();
            match op {
                Op::Reshape(x) => {
                    let gx = Self::reshape_grad(gy, x.shape().to_vec())?;
                    grads.add(&x, gx)?;
                }
            }
        }

        Ok(grads)
    }

    fn reshape_grad(gy: &Tensor<T>, shape: Vec<usize>) -> Result<Tensor<T>> {
        gy.reshape(shape)
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

impl<T: Num> ops::Add<Tensor<T>> for Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn add(self, rhs: Tensor<T>) -> Result<Tensor<T>> {
        self.add_impl(&rhs)
    }
}

impl<T: Num> ops::Add<Tensor<T>> for Result<Tensor<T>> {
    type Output = Result<Tensor<T>>;

    fn add(self, rhs: Tensor<T>) -> Result<Tensor<T>> {
        self?.add_impl(&rhs)
    }
}
