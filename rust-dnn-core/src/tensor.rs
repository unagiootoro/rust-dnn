use std::{
    cell::RefCell,
    collections::HashSet,
    marker::PhantomData,
    ops::{self, Deref, Range},
    rc::Rc,
};

#[cfg(feature = "cuda")]
use rust_dnn_cuda_kernel::gpu_buffer::GPUBuffer;

use crate::{
    backend::Backend,
    device::{Device, DeviceInfo},
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

pub struct TensorState<B: Backend, T: Num> {
    id: usize,
    storage: Rc<RefCell<Storage<T>>>,
    layout: Layout,
    device: Device<B>,
    dtype: DType,
    is_requires_grad: bool,
    op: Option<Op<B, T>>,
    _marker: PhantomData<(B, T)>,
}

#[derive(Clone)]
pub struct Tensor<B: Backend, T: Num>(Rc<TensorState<B, T>>);

impl<B: Backend, T: Num> Deref for Tensor<B, T> {
    type Target = TensorState<B, T>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl<B: Backend, T: Num> Tensor<B, T> {
    pub fn new(
        storage: Rc<RefCell<Storage<T>>>,
        layout: Layout,
        device: Device<B>,
        dtype: DType,
        requires_grad: bool,
        op: Option<Op<B, T>>,
    ) -> Self {
        let id = TENSOR_ID_COUNTER.with(|counter| *counter.borrow());
        TENSOR_ID_COUNTER.with(|counter| *counter.borrow_mut() = id + 1);
        let state = TensorState::<B, T> {
            id,
            storage,
            layout,
            device,
            dtype,
            is_requires_grad: requires_grad,
            op,
            _marker: PhantomData,
        };
        Self(Rc::new(state))
    }

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>, device: Device<B>) -> Result<Self> {
        let len = Self::compute_len(&shape);
        if data.len() != len {
            let msg = format!(
                "Invalid data length(data len = {}, shape len = {})",
                data.len(),
                len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let storage = match *device.info() {
            DeviceInfo::Cpu => Storage::CpuStorage(data),
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => Storage::CudaStorage(GPUBuffer::from_vec(&data)),
        };
        let stride = Self::compute_stride(&shape);
        let layout = Layout::new(shape, stride, 0);
        Ok(Self::new(
            Rc::new(RefCell::new(storage)),
            layout,
            device,
            T::dtype(),
            false,
            None,
        ))
    }

    pub fn from_scalar(value: T, device: Device<B>) -> Self {
        let shape = vec![1];
        let data = vec![value];
        let storage = match *device.info() {
            DeviceInfo::Cpu => Storage::CpuStorage(data),
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => Storage::CudaStorage(GPUBuffer::from_vec(&data)),
        };
        let stride = Self::compute_stride(&shape);
        let layout = Layout::new(shape, stride, 0);
        Self::new(
            Rc::new(RefCell::new(storage)),
            layout,
            device,
            T::dtype(),
            false,
            None,
        )
    }

    pub fn zeros(shape: Vec<usize>, device: Device<B>) -> Self {
        Self::fill(shape, T::zero(), device)
    }

    pub fn ones(shape: Vec<usize>, device: Device<B>) -> Self {
        Self::fill(shape, T::one(), device)
    }

    pub fn fill(shape: Vec<usize>, value: T, device: Device<B>) -> Self {
        let mut data = Vec::new();
        for _ in 0..Self::compute_len(&shape) {
            data.push(value);
        }
        let stride = Self::compute_stride(&shape);
        let storage = match *device.info() {
            DeviceInfo::Cpu => Storage::CpuStorage(data),
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => Storage::CudaStorage(GPUBuffer::from_vec(&data)),
        };
        let layout = Layout::new(shape, stride, 0);
        Self::new(
            Rc::new(RefCell::new(storage)),
            layout,
            device,
            T::dtype(),
            false,
            None,
        )
    }

    pub fn arange(range: Range<isize>, device: Device<B>) -> Self {
        let mut data = Vec::new();
        for i in range {
            data.push(T::from_isize(i));
        }
        let shape = vec![data.len()];
        let stride = Self::compute_stride(&shape);
        let storage = match *device.info() {
            DeviceInfo::Cpu => Storage::CpuStorage(data),
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => Storage::CudaStorage(GPUBuffer::from_vec(&data)),
        };
        let layout = Layout::new(shape, stride, 0);
        Self::new(
            Rc::new(RefCell::new(storage)),
            layout,
            device,
            T::dtype(),
            false,
            None,
        )
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

    pub fn to_vec(&self) -> Result<Vec<T>> {
        let x = self.contiguous()?;
        let storage = &*x.storage.borrow();
        Ok(storage.to_vec_range(x.layout.storage_offset()..x.layout.len()))
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

    pub fn requires_grad(&self) -> Self {
        Self::new(
            self.storage.clone(),
            self.layout.clone(),
            self.device.clone(),
            self.dtype,
            true,
            self.op.clone(),
        )
    }

    pub fn detach(&self) -> Self {
        if !self.is_requires_grad() && self.op.is_none() {
            self.clone()
        } else {
            Self::new(
                self.storage.clone(),
                self.layout.clone(),
                self.device.clone(),
                self.dtype,
                false,
                None,
            )
        }
    }

    fn add_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        self.op2_impl(rhs, Self::op_add, |t1, t2| Op::Add(t1, t2))
    }

    fn sub_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        self.op2_impl(rhs, Self::op_sub, |t1, t2| Op::Sub(t1, t2))
    }

    fn mul_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        self.op2_impl(rhs, Self::op_mul, |t1, t2| Op::Mul(t1, t2))
    }

    fn div_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        self.op2_impl(rhs, Self::op_div, |t1, t2| Op::Div(t1, t2))
    }

    fn op_add(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        B::op_add::<T>(lhs_storage, rhs_storage, lhs_layout, rhs_layout)
    }

    fn op_sub(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        B::op_sub::<T>(lhs_storage, rhs_storage, lhs_layout, rhs_layout)
    }

    fn op_mul(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        B::op_mul::<T>(lhs_storage, rhs_storage, lhs_layout, rhs_layout)
    }

    fn op_div(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        B::op_div::<T>(lhs_storage, rhs_storage, lhs_layout, rhs_layout)
    }

    fn op1_impl<F1, F2>(&self, f1: F1, f2: F2) -> Result<Self>
    where
        F1: for<'a> Fn(&'a Storage<T>, &'a Layout) -> Result<Storage<T>>,
        F2: Fn(Tensor<B, T>) -> Op<B, T>,
    {
        let input_storage = &*self.storage.borrow();

        let output_storage = f1(input_storage, &self.layout)?;

        let op = if self.is_requires_grad {
            Some(f2(self.clone()))
        } else {
            None
        };
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            self.layout.clone(),
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Result::Ok(output)
    }

    fn op2_impl<F1, F2>(&self, rhs: &Tensor<B, T>, f1: F1, f2: F2) -> Result<Self>
    where
        F1: for<'a> Fn(
            &'a Storage<T>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
        ) -> Result<Storage<T>>,
        F2: Fn(Tensor<B, T>, Tensor<B, T>) -> Op<B, T>,
    {
        let is_requires_grad = self.is_requires_grad() || rhs.is_requires_grad();
        let op = if is_requires_grad {
            Some(f2(self.clone(), rhs.clone()))
        } else {
            None
        };

        let broadcasted_shape = Self::broadcast_shape(self.shape(), rhs.shape())?;
        let lhs = self.detach().broadcast_to(broadcasted_shape.clone())?;
        let rhs = rhs.detach().broadcast_to(broadcasted_shape.clone())?;
        let lhs_storage = &*lhs.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();

        let output_storage = f1(lhs_storage, rhs_storage, &lhs.layout, &rhs.layout)?;

        let stride = Self::compute_stride(&broadcasted_shape);
        let layout = Layout::new(broadcasted_shape, stride, 0);
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            is_requires_grad,
            op,
        );
        Result::Ok(output)
    }

    fn op2_scalar_impl<F1, F2>(&self, rhs: T, f1: F1, f2: F2) -> Result<Self>
    where
        F1: for<'a> Fn(&'a Storage<T>, &'a Layout, T) -> Result<Storage<T>>,
        F2: Fn(Tensor<B, T>, T) -> Op<B, T>,
    {
        let input_storage = &*self.storage.borrow();

        let output_storage = f1(input_storage, &self.layout, rhs)?;

        let op = if self.is_requires_grad {
            Some(f2(self.clone(), rhs))
        } else {
            None
        };
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            self.layout.clone(),
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Result::Ok(output)
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
            self.device.clone(),
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
        let op = if self.is_requires_grad {
            Some(Op::PermutedAxes(self.clone(), axes.to_vec()))
        } else {
            None
        };
        let output = Tensor::new(
            self.storage.clone(),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
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
        let op = if self.is_requires_grad {
            Some(Op::BroadcastTo(self.clone()))
        } else {
            None
        };
        let output = Tensor::new(
            Rc::clone(&input.storage),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Ok(output)
    }

    pub fn sum_to(&self, shape: &[usize]) -> Result<Tensor<B, T>> {
        let y = if self.shape() == shape {
            self.clone()
        } else {
            let mut shape2 = Vec::new();
            if shape.len() < self.shape().len() {
                for _ in 0..(self.shape().len() - shape.len()) {
                    shape2.push(1);
                }
            }
            for dim in shape {
                shape2.push(*dim);
            }

            for (i, dim) in self.shape().iter().enumerate() {
                if shape2[i] != *dim && shape2[i] != 1 && *dim != 1 {
                    return Err(Error::ArgumentsError {
                        msg: format!(
                            "Invalud sum_to input shape: self.shape = {:?}, target shape = {:?}",
                            self.shape(),
                            shape
                        ),
                    });
                }
            }

            let mut result = self.clone();
            for i in 0..self.ndim() {
                if self.shape()[i] > 1 && shape2[i] == 1 {
                    result = result.sum_axis(i, true)?;
                }
            }
            result.reshape(shape.to_vec())?
        };
        Ok(y)
    }

    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> Result<Tensor<B, T>> {
        self.op_reduce_axis_impl(axis, keepdims, Self::op_sum_axis, |t, axis, keepdims| {
            Op::SumAxis(t, axis, keepdims)
        })
    }

    pub fn op_sum_axis(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        B::sum_axis::<T>(input_storage, input_layout, output_layout, axis)
    }

    fn op_reduce_axis_impl<F1, F2>(
        &self,
        axis: usize,
        keepdims: bool,
        f1: F1,
        f2: F2,
    ) -> Result<Self>
    where
        F1: for<'a> Fn(&'a Storage<T>, &'a Layout, &'a Layout, usize) -> Result<Storage<T>>,
        F2: Fn(Tensor<B, T>, usize, bool) -> Op<B, T>,
    {
        let mut output_shape = Vec::new();
        for i in 0..self.ndim() {
            if i == axis {
                output_shape.push(1);
            } else {
                output_shape.push(self.shape()[i]);
            }
        }

        let mut output_data = Vec::new();
        let output_data_len = Self::compute_len(&output_shape);
        for _ in 0..output_data_len {
            output_data.push(T::zero());
        }

        let storage = &*self.storage.borrow();

        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape.clone(), output_stride, 0);

        let output_storage = f1(storage, &self.layout, &output_layout, axis)?;

        let output_shape = if keepdims {
            output_shape
        } else {
            let mut output_shape2 = Vec::new();
            for i in 0..output_shape.len() {
                if i != axis {
                    output_shape2.push(output_shape[i]);
                }
            }
            output_shape2
        };
        let output_stride = Self::compute_stride(&output_shape);

        let output_layout = Layout::new(output_shape, output_stride, 0);
        let op = if self.is_requires_grad {
            Some(f2(self.clone(), axis, keepdims))
        } else {
            None
        };
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Result::Ok(output)
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
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let input_storage = &*self.storage.borrow();
        let output_storage = B::contiguous::<T>(input_storage, &self.layout)?;

        let layout = Layout::new(
            self.shape().to_vec(),
            Self::compute_stride(&self.shape()),
            0,
        );
        let op = if self.is_requires_grad {
            Some(Op::Contiguous(self.clone()))
        } else {
            None
        };
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Ok(output)
    }

    #[cfg(feature = "cuda")]
    pub fn to_device<B2: Backend>(&self, device: Device<B2>) -> Result<Tensor<B2, T>> {
        if self.device.info() == device.info() {
            return unsafe { Ok(self.detach().reinterpret_cast_backend::<B2>()) };
        } else if *self.device.info() == DeviceInfo::Cpu && *device.info() == DeviceInfo::Cuda {
            let storage = self.storage.borrow();
            let cpu_storage = storage.get_cpu_storage()?;
            let gpu_buffer = GPUBuffer::from_vec(cpu_storage);
            let output_storage = Storage::CudaStorage(gpu_buffer);
            let output = Tensor::new(
                Rc::new(RefCell::new(output_storage)),
                self.layout.clone(),
                device,
                self.dtype,
                self.is_requires_grad,
                None,
            );
            return Ok(output);
        } else if *self.device.info() == DeviceInfo::Cuda && *device.info() == DeviceInfo::Cpu {
            let storage = self.storage.borrow();
            let cuda_storage = storage.get_cuda_storage()?;
            let output_storage = Storage::CpuStorage(cuda_storage.to_vec());
            let output = Tensor::new(
                Rc::new(RefCell::new(output_storage)),
                self.layout.clone(),
                device,
                self.dtype,
                self.is_requires_grad,
                None,
            );
            return Ok(output);
        }
        Err(Error::ArgumentsError {
            msg: format!("Invalid device(device = {:?}", device.info()).to_string(),
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn to_device<B2: Backend>(&self, _device: Device<B2>) -> Tensor<B2, T> {
        return unsafe { self.detach().reinterpret_cast_backend::<B2>() };
    }

    pub unsafe fn reinterpret_cast_backend<B2: Backend>(self) -> Tensor<B2, T> {
        unsafe { std::mem::transmute::<Self, Tensor<B2, T>>(self) }
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

impl<B: Backend, T: Float> Tensor<B, T> {
    fn neg_impl(&self) -> Result<Self> {
        self.op1_impl(Self::op_neg, |t| Op::Neg(t))
    }

    fn op_neg(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        B::op_neg::<T>(storage, layout)
    }

    pub fn pow_scalar(&self, rhs: T) -> Result<Self> {
        self.op2_scalar_impl(rhs, Self::op_pow_scalar, |t1, t2| Op::PowScalar(t1, t2))
    }

    fn op_pow_scalar(storage: &Storage<T>, layout: &Layout, rhs: T) -> Result<Storage<T>> {
        B::op_pow_scalar::<T>(storage, layout, rhs)
    }
}

impl<B: Backend, T: Float> Tensor<B, T> {
    pub fn sorted_nodes(&self) -> Vec<Tensor<B, T>> {
        let mut nodes: Vec<(usize, Tensor<B, T>)> = Vec::new();

        let mut seen_set = HashSet::new();
        let mut work_nodes = Vec::<(usize, Tensor<B, T>)>::new();
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
                Op::BroadcastTo(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::PermutedAxes(x, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Contiguous(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::SumAxis(x, _, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Add(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Sub(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Mul(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Div(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Neg(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::PowScalar(x, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
            }
        }

        nodes.sort_by(|a, b| a.0.cmp(&b.0));
        nodes.into_iter().map(|a| a.1).collect()
    }

    fn add_work_node(
        depth: usize,
        node: Tensor<B, T>,
        work_nodes: &mut Vec<(usize, Tensor<B, T>)>,
        seen_set: &mut HashSet<usize>,
    ) {
        if !seen_set.contains(&node.id()) {
            seen_set.insert(node.id());
            work_nodes.push((depth, node.clone()));
        }
    }

    pub fn backward(&self) -> Result<Gradients<B, T>> {
        let mut grads = Gradients::new();
        if !self.is_requires_grad {
            return Ok(grads);
        }

        let gy = Tensor::<B, T>::ones(self.shape().to_vec(), self.device.clone());
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
                Op::BroadcastTo(x) => {
                    let gx = Self::broadcast_to_grad(gy, x.shape().to_vec())?;
                    grads.add(&x, gx)?;
                }
                Op::PermutedAxes(x, axes) => {
                    let gx = Self::permuted_axes_grad(gy, axes)?;
                    grads.add(&x, gx)?;
                }
                Op::Contiguous(x) => {
                    let gx = Self::contiguous_grad(gy);
                    grads.add(&x, gx)?;
                }
                Op::SumAxis(x, axis, keepdims) => {
                    let gx = Self::sum_axis_grad(gy, x.shape(), axis, keepdims)?;
                    grads.add(&x, gx)?;
                }
                Op::Add(x1, x2) => {
                    let (gx1, gx2) = Self::add_grad(gy, x1.shape(), x2.shape())?;
                    grads.add(&x1, gx1)?;
                    grads.add(&x2, gx2)?;
                }
                Op::Sub(x1, x2) => {
                    let (gx1, gx2) = Self::sub_grad(gy, x1.shape(), x2.shape())?;
                    grads.add(&x1, gx1)?;
                    grads.add(&x2, gx2)?;
                }
                Op::Mul(x1, x2) => {
                    let (gx1, gx2) = Self::mul_grad(gy, &x1, &x2)?;
                    grads.add(&x1, gx1)?;
                    grads.add(&x2, gx2)?;
                }
                Op::Div(x1, x2) => {
                    let (gx1, gx2) = Self::div_grad(gy, &x1, &x2)?;
                    grads.add(&x1, gx1)?;
                    grads.add(&x2, gx2)?;
                }
                Op::Neg(x) => {
                    let gx = Self::neg_grad(gy)?;
                    grads.add(&x, gx)?;
                }
                Op::PowScalar(x, rhs) => {
                    let gx = Self::pow_scalar_grad(gy, &x, rhs)?;
                    grads.add(&x, gx)?;
                }
            }
            grads.remove(&node);
        }

        Ok(grads)
    }

    fn reshape_grad(gy: &Tensor<B, T>, shape: Vec<usize>) -> Result<Tensor<B, T>> {
        gy.reshape(shape)
    }

    fn broadcast_to_grad(gy: &Tensor<B, T>, shape: Vec<usize>) -> Result<Tensor<B, T>> {
        gy.sum_to(&shape)
    }

    fn permuted_axes_grad(gy: &Tensor<B, T>, axes: Vec<usize>) -> Result<Tensor<B, T>> {
        if gy.ndim() != axes.len() {
            return Err(Error::ArgumentsError {
                msg: format!(
                    "Mismatch dims(gy.ndim() = {}, self.axes.len() = {})",
                    gy.ndim(),
                    axes.len()
                ),
            });
        }
        let mut inv_axes = Vec::new();
        for i in 0..gy.ndim() {
            let position = axes.iter().position(|&axis| axis == i).unwrap();
            inv_axes.push(position);
        }
        gy.permuted_axes(&inv_axes)
    }

    fn contiguous_grad(gy: &Tensor<B, T>) -> Tensor<B, T> {
        gy.clone()
    }

    fn sum_axis_grad(
        gy: &Tensor<B, T>,
        x_shape: &[usize],
        axis: usize,
        keepdims: bool,
    ) -> Result<Tensor<B, T>> {
        let gy = if keepdims {
            gy.clone()
        } else {
            let mut output_shape = Vec::new();
            let mut j = 0;
            for i in 0..x_shape.len() {
                if i == axis {
                    output_shape.push(1);
                } else {
                    output_shape.push(gy.shape()[j]);
                    j += 1;
                }
            }
            gy.reshape(output_shape)?
        };
        gy.broadcast_to(x_shape.to_vec())
    }

    fn add_grad(
        gy: &Tensor<B, T>,
        x1_shape: &[usize],
        x2_shape: &[usize],
    ) -> Result<(Tensor<B, T>, Tensor<B, T>)> {
        let gx1 = gy.sum_to(x1_shape)?;
        let gx2 = gy.sum_to(x2_shape)?;
        Ok((gx1, gx2))
    }

    fn sub_grad(
        gy: &Tensor<B, T>,
        x1_shape: &[usize],
        x2_shape: &[usize],
    ) -> Result<(Tensor<B, T>, Tensor<B, T>)> {
        let gx1 = gy.sum_to(x1_shape)?;
        let gx2 = (-gy)?.sum_to(x2_shape)?;
        Ok((gx1, gx2))
    }

    fn mul_grad(
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<(Tensor<B, T>, Tensor<B, T>)> {
        let gx1 = (gy * x2)?.sum_to(x1.shape())?;
        let gx2 = (gy * x1)?.sum_to(x2.shape())?;
        Ok((gx1, gx2))
    }

    fn div_grad(
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<(Tensor<B, T>, Tensor<B, T>)> {
        let gx1 = (gy / x2)?.sum_to(x1.shape())?;
        let gx2 = (gy * (-x1 / x2.pow_scalar(T::from_f32(2.0))?)?)?.sum_to(x2.shape())?;
        Ok((gx1, gx2))
    }

    fn neg_grad(gy: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        -gy
    }

    fn pow_scalar_grad(gy: &Tensor<B, T>, x: &Tensor<B, T>, rhs: T) -> Result<Tensor<B, T>> {
        let rhs_tensor = Tensor::from_scalar(rhs, x.device.clone());
        rhs_tensor * x.pow_scalar(rhs - T::from_f32(1.0))? * gy
    }
}

#[macro_export]
macro_rules! ten {
    ( $( [ $($inner:tt)* ] ),+ $(,)? ) => {{
        let arrays = vec![$( ten![ $($inner)* ] ),+];
        let shape = {
            let mut s = arrays[0].shape().to_vec();
            s.insert(0, arrays.len());
            s
        };
        let data = arrays
            .into_iter()
            .flat_map(|a| a.to_vec().unwrap())
            .collect::<Vec<_>>();
        tensor::Tensor::from_vec(data, shape, Device::get_cpu_device()).unwrap()
    }};
    ( $($x:expr),+ $(,)? ) => {{
        let data = vec![$($x),+];
        let shape = vec![data.len()];
        tensor::Tensor::from_vec(data, shape, Device::get_cpu_device()).unwrap()
    }};
}

macro_rules! define_op_arg2 {
    ($op_name:ident, $fn_name:ident, $impl_fn_name:ident) => {
        impl<B: Backend, T: Num> std::ops::$op_name<Tensor<B, T>> for Tensor<B, T> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Result<Tensor<B, T>> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<Tensor<B, T>> for Result<Tensor<B, T>> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Result<Tensor<B, T>> {
                self?.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<&Tensor<B, T>> for Tensor<B, T> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Result<Tensor<B, T>> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<&Tensor<B, T>> for Result<Tensor<B, T>> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Result<Tensor<B, T>> {
                self?.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<Tensor<B, T>> for &Tensor<B, T> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Result<Tensor<B, T>> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<&Tensor<B, T>> for &Tensor<B, T> {
            type Output = Result<Tensor<B, T>>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Result<Tensor<B, T>> {
                self.$impl_fn_name(&rhs)
            }
        }
    };
}

define_op_arg2!(Add, add, add_impl);
define_op_arg2!(Sub, sub, sub_impl);
define_op_arg2!(Mul, mul, mul_impl);
define_op_arg2!(Div, div, div_impl);

impl<B: Backend, T: Float> ops::Neg for Tensor<B, T> {
    type Output = Result<Tensor<B, T>>;

    fn neg(self) -> Result<Tensor<B, T>> {
        self.neg_impl()
    }
}

impl<B: Backend, T: Float> ops::Neg for &Tensor<B, T> {
    type Output = Result<Tensor<B, T>>;

    fn neg(self) -> Result<Tensor<B, T>> {
        self.neg_impl()
    }
}
