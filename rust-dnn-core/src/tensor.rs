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

    pub fn to_vec(&self) -> Vec<T> {
        let x = self.contiguous();
        let storage = &*x.storage.borrow();
        let start = x.layout.storage_offset();
        let end = start + x.layout.len();
        storage.to_vec_range(start..end)
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

    pub fn device(&self) -> Device<B> {
        self.device
    }

    pub fn with_op(&self, op: Option<Op<B, T>>) -> Self {
        Self::new(
            self.storage.clone(),
            self.layout.clone(),
            self.device.clone(),
            self.dtype,
            true,
            op,
        )
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
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Add(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_add)
    }

    pub fn add_assign(&self, rhs: &Tensor<B, T>) -> Result<()> {
        self.op2_inplace_impl(rhs, B::op_add_assign)
    }

    fn sub_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Sub(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_sub)
    }

    pub fn sub_assign(&self, rhs: &Tensor<B, T>) -> Result<()> {
        self.op2_inplace_impl(rhs, B::op_sub_assign)
    }

    fn mul_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        let op: Option<Op<B, T>> = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Mul(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_mul)
    }

    pub fn mul_assign(&self, rhs: &Tensor<B, T>) -> Result<()> {
        self.op2_inplace_impl(rhs, B::op_mul_assign)
    }

    fn div_impl(&self, rhs: &Tensor<B, T>) -> Result<Self> {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Div(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_div)
    }

    pub fn div_assign(&self, rhs: &Tensor<B, T>) -> Result<()> {
        self.op2_inplace_impl(rhs, B::op_div_assign)
    }

    pub fn eq(&self, rhs: &Tensor<B, T>) -> Result<Tensor<B, u32>> {
        self.op2_impl(rhs, None, B::eq)
    }

    pub fn lt(&self, rhs: &Tensor<B, T>) -> Result<Tensor<B, u32>> {
        self.op2_impl(rhs, None, B::lt)
    }

    pub fn le(&self, rhs: &Tensor<B, T>) -> Result<Tensor<B, u32>> {
        self.op2_impl(rhs, None, B::le)
    }

    pub fn gt(&self, rhs: &Tensor<B, T>) -> Result<Tensor<B, u32>> {
        self.op2_impl(rhs, None, B::gt)
    }

    pub fn ge(&self, rhs: &Tensor<B, T>) -> Result<Tensor<B, u32>> {
        self.op2_impl(rhs, None, B::ge)
    }

    fn op1_impl<F>(&self, op: Option<Op<B, T>>, f: F) -> Self
    where
        F: for<'a> Fn(&'a Storage<T>, &'a Layout) -> Result<Storage<T>>,
    {
        let input = self.contiguous();
        let input_storage = &*input.storage.borrow();
        let output_storage = f(input_storage, &input.layout).unwrap();
        let ouput_layout = Layout::new(input.shape().to_vec(), input.stride().to_vec(), 0);
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            ouput_layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        output
    }

    fn op2_impl<T2: Num, F>(
        &self,
        rhs: &Tensor<B, T>,
        op: Option<Op<B, T2>>,
        f: F,
    ) -> Result<Tensor<B, T2>>
    where
        F: for<'a> Fn(
            &'a Storage<T>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
        ) -> Result<Storage<T2>>,
    {
        let broadcasted_shape = Self::broadcast_shape(self.shape(), rhs.shape())?;
        let lhs = self.detach().broadcast_to(broadcasted_shape.clone())?;
        let rhs = rhs.detach().broadcast_to(broadcasted_shape.clone())?;
        let lhs_storage = &*lhs.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();

        let output_storage = f(lhs_storage, rhs_storage, &lhs.layout, &rhs.layout)?;

        let stride = Self::compute_stride(&broadcasted_shape);
        let layout = Layout::new(broadcasted_shape, stride, 0);
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad() || rhs.is_requires_grad(),
            op,
        );
        Ok(output)
    }

    fn op2_inplace_impl<F>(&self, rhs: &Tensor<B, T>, f: F) -> Result<()>
    where
        F: for<'a> Fn(&'a mut Storage<T>, &'a Storage<T>, &'a Layout, &'a Layout) -> Result<()>,
    {
        let broadcasted_shape = Self::broadcast_shape(self.shape(), rhs.shape())?;
        if self.shape() != broadcasted_shape {
            return Err(Error::ArgumentsError {
                msg: format!(
                    "Invalid shape(self.shape = {:?}, broadcasted_shape = {:?})",
                    self.shape(),
                    broadcasted_shape
                ),
            });
        }
        let rhs = rhs.detach().broadcast_to(broadcasted_shape.clone())?;
        let lhs_storage = &mut *self.storage.borrow_mut();
        let rhs_storage = &*rhs.storage.borrow();

        f(lhs_storage, rhs_storage, &self.layout, &rhs.layout)
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let output_len = Self::compute_len(&shape);
        if self.len() != output_len {
            let msg = format!(
                "Length mismatch (from shape = {:?}, from shape len = {}, to shape = {:?}, to shape len = {})",
                self.shape(),
                self.len(),
                shape,
                output_len
            );
            return Err(Error::ArgumentsError { msg });
        }
        let input = self.contiguous();
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
        Ok(output)
    }

    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::new();
        for dim in self.shape() {
            if *dim != 1 {
                shape.push(*dim);
            }
        }
        self.reshape(shape).unwrap()
    }

    pub fn squeeze_axes(&self, axes: &[usize]) -> Result<Self> {
        for axis in axes {
            if *axis > self.ndim() {
                return Err(Error::ArgumentsError {
                    msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
                });
            }
        }
        let mut shape = Vec::new();
        for (axis, dim) in self.shape().iter().enumerate() {
            if axes.contains(&axis) {
                if *dim != 1 {
                    return Err(Error::ArgumentsError {
                        msg: format!("Dim is not 1(dim = {}, axis = {})", dim, axis),
                    });
                }
            } else {
                shape.push(self.shape()[axis]);
            }
        }
        Ok(self.reshape(shape).unwrap())
    }

    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
            });
        }
        let mut shape = self.shape().to_vec();
        shape.insert(axis, 1);
        Ok(self.reshape(shape).unwrap())
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
        Ok(output)
    }

    pub fn reversed_axes(&self) -> Result<Self> {
        let mut axes = Vec::new();
        for axis in (0..self.ndim()).rev() {
            axes.push(axis);
        }
        self.permuted_axes(&axes)
    }

    pub fn get_item(&self, ranges: Vec<(usize, usize)>) -> Result<Self> {
        for (i, range) in ranges.iter().enumerate() {
            if range.1 > self.shape()[i] {
                return Err(Error::ArgumentsError {
                    msg: format!(
                        "Invalid range(self.shape = {:?}, ranges = {:?})",
                        self.shape(),
                        ranges
                    ),
                });
            }
        }

        let mut output_shape = Vec::new();
        for range in &ranges {
            output_shape.push(range.1 - range.0);
        }
        let storage_offset =
            Self::compute_offset_by_ranges(self.storage_offset(), self.stride(), &ranges);
        let output_layout = Layout::new(output_shape, self.stride().to_vec(), storage_offset);
        let output = Tensor::new(
            Rc::clone(&self.storage),
            output_layout,
            self.device,
            self.dtype,
            self.is_requires_grad(),
            Some(Op::GetItem(self.clone(), ranges)),
        );
        Ok(output)
    }

    pub fn set_item(&self, ranges: &Vec<(usize, usize)>, src: &Self) -> Result<()> {
        let x = self.get_item(ranges.clone())?;
        x.copy(src)
    }

    pub fn select(&self, axis: usize, index: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
            });
        }
        let mut ranges = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                if index >= *dim {
                    return Err(Error::ArgumentsError {
                        msg: format!(
                            "Invalud index: self.shape = {:?}, axis = {}, index = {}",
                            self.shape(),
                            axis,
                            index,
                        ),
                    });
                }
                ranges.push((index, index + 1));
            } else {
                ranges.push((0, *dim));
            }
        }
        Ok(self
            .get_item(ranges)
            .unwrap()
            .squeeze_axes(&[axis])
            .unwrap())
    }

    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self> {
        if axis > self.ndim() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
            });
        }
        let mut ranges = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                if start + length >= *dim {
                    return Err(Error::ArgumentsError {
                        msg: format!(
                            "Invalud index: self.shape = {:?}, axis = {}, start = {}, length = {}",
                            self.shape(),
                            axis,
                            start,
                            length,
                        ),
                    });
                }
                ranges.push((start, start + length));
            } else {
                ranges.push((0, *dim));
            }
        }
        Ok(self.get_item(ranges).unwrap())
    }

    pub fn cat(tensors: &[Self], axis: usize) -> Result<Self> {
        let mut split_sections = Vec::new();

        let mut output_shape = Vec::new();
        for i in 0..tensors[0].ndim() {
            if i == axis {
                let mut dim = 0;
                for x in tensors {
                    dim += x.shape()[i];
                }
                output_shape.push(dim);
            } else {
                output_shape.push(tensors[0].shape()[i]);
            }
        }

        let y = Tensor::zeros(output_shape, tensors[0].device());
        let y = if tensors.iter().any(|x| x.is_requires_grad()) {
            y.requires_grad()
        } else {
            y
        };

        let mut total_axis_ndim = 0;
        for x in tensors {
            let mut ranges = Vec::new();
            for i in 0..y.shape().len() {
                if i == axis {
                    let next_total_axis_ndim = total_axis_ndim + x.shape()[i];
                    ranges.push((total_axis_ndim, next_total_axis_ndim));
                    total_axis_ndim = next_total_axis_ndim;
                    split_sections.push(x.shape()[i]);
                } else {
                    ranges.push((0, x.shape()[i]));
                }
            }
            y.set_item(&ranges, &x)?
        }

        let y = y.with_op(Some(Op::Cat(tensors.to_vec(), axis, split_sections)));
        Ok(y)
    }

    pub fn split(&self, axis: usize, split_sections: &[usize]) -> Result<Vec<Self>> {
        let mut ys = Vec::new();
        let mut total_axis_ndim = 0;
        for dim in split_sections {
            let mut ranges = Vec::new();
            for i in 0..self.ndim() {
                if i == axis {
                    let next_total_axis_ndim = total_axis_ndim + dim;
                    ranges.push((total_axis_ndim, next_total_axis_ndim));
                    total_axis_ndim = next_total_axis_ndim;
                } else {
                    ranges.push((0, self.shape()[i]));
                }
            }
            let y = self.get_item(ranges)?;
            ys.push(y);
        }
        Ok(ys)
    }

    pub fn copy(&self, src: &Self) -> Result<()> {
        self.op2_inplace_impl(src, B::copy)
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

    pub fn sum(&self) -> Result<Tensor<B, T>> {
        let op = if self.is_requires_grad() {
            Some(Op::Sum(self.clone()))
        } else {
            None
        };
        self.op_reduce_impl(op, B::sum)
    }

    pub fn sum_axis(&self, axis: usize, keepdims: bool) -> Result<Tensor<B, T>> {
        let op = if self.is_requires_grad() {
            Some(Op::SumAxis(self.clone()))
        } else {
            None
        };
        self.op_reduce_axis_impl(axis, keepdims, op, B::sum_axis)
    }

    pub fn mean(&self) -> Result<Self> {
        self.sum() / Tensor::from_scalar(T::from_usize(self.len()), self.device)
    }

    pub fn mean_axis(&self, axis: usize, keepdims: bool) -> Result<Tensor<B, T>> {
        self.sum_axis(axis, keepdims)
            / Tensor::from_scalar(T::from_usize(self.shape()[axis]), self.device)
    }

    pub fn max(&self) -> Result<Tensor<B, T>> {
        let output = self.op_reduce_impl(None, B::max)?;
        let op = if self.is_requires_grad() {
            Some(Op::Max(self.clone(), output.detach()))
        } else {
            None
        };
        Ok(output.with_op(op))
    }

    pub fn max_axis(&self, axis: usize, keepdims: bool) -> Result<Tensor<B, T>> {
        let output = self.op_reduce_axis_impl(axis, keepdims, None, B::max_axis)?;
        let op = if self.is_requires_grad() {
            Some(Op::MaxAxis(self.clone(), output.detach()))
        } else {
            None
        };
        Ok(output.with_op(op))
    }

    fn op_reduce_impl<F>(&self, op: Option<Op<B, T>>, f: F) -> Result<Self>
    where
        F: for<'a> Fn(&'a Storage<T>, &'a Layout) -> Result<Storage<T>>,
    {
        let storage = &*self.storage.borrow();
        let output_storage = f(storage, &self.layout)?;
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            Layout::new(vec![1], vec![1], 0),
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
        Ok(output)
    }

    fn op_reduce_axis_impl<F>(
        &self,
        axis: usize,
        keepdims: bool,
        op: Option<Op<B, T>>,
        f: F,
    ) -> Result<Self>
    where
        F: for<'a> Fn(&'a Storage<T>, &'a Layout, &'a Layout, usize) -> Result<Storage<T>>,
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

        let output_storage = f(storage, &self.layout, &output_layout, axis)?;
        let output_stride = Self::compute_stride(&output_shape);

        let output_layout = Layout::new(output_shape, output_stride, 0);
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );

        if keepdims {
            Ok(output)
        } else {
            Ok(output.squeeze_axes(&[axis]).unwrap())
        }
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

    fn compute_offset_by_ranges(
        storage_offset: usize,
        stride: &[usize],
        ranges: &[(usize, usize)],
    ) -> usize {
        let mut offset = 0;
        for i in 0..stride.len() {
            if stride[i] > 0 {
                offset += ranges[i].0 * stride[i];
            }
        }
        storage_offset + offset
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

    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }

        let input_storage = &*self.storage.borrow();
        let output_storage = B::contiguous::<T>(input_storage, &self.layout).unwrap();

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
        output
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
    pub fn to_device<B2: Backend>(&self, _device: Device<B2>) -> Result<Tensor<B2, T>> {
        return unsafe { Ok(self.detach().reinterpret_cast_backend::<B2>()) };
    }

    pub unsafe fn reinterpret_cast_backend<B2: Backend>(self) -> Tensor<B2, T> {
        unsafe { std::mem::transmute::<Self, Tensor<B2, T>>(self) }
    }

    pub fn to_dtype<T2: Num>(&self) -> Result<Tensor<B, T2>> {
        let storage = self.storage.borrow().to_dtype();
        let output = Tensor::new(
            Rc::new(RefCell::new(storage)),
            self.layout.clone(),
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            None,
        );
        Ok(output)
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
    fn neg_impl(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Neg(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::op_neg)
    }

    fn matmul2d(&self, rhs: &Self) -> Result<Self> {
        let lhs_storage = &*self.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();
        let output_storage = B::matmul(lhs_storage, &rhs_storage, &self.layout, &rhs.layout)?;
        let lhs_rows = self.shape()[0];
        let rhs_cols = rhs.shape()[1];
        let output_shape = vec![lhs_rows, rhs_cols];
        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape, output_stride, 0);
        let output = Self::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device,
            self.dtype,
            self.is_requires_grad() || rhs.is_requires_grad(),
            None,
        );
        Ok(output)
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let mut batch_a_shape = Vec::new();
        for i in 0..self.ndim() - 2 {
            batch_a_shape.push(self.shape()[i]);
        }

        let mut a_batch_size = 1;
        for i in 0..self.ndim() - 2 {
            a_batch_size *= self.shape()[i];
        }

        let mut b_batch_size = 1;
        for i in 0..rhs.ndim() - 2 {
            b_batch_size *= rhs.shape()[i];
        }

        if a_batch_size > 1 && b_batch_size > 1 && a_batch_size != b_batch_size {
            panic!(
                "a_batch_size = {}, b_batch_size = {}",
                a_batch_size, b_batch_size
            );
        }
        let batch_size = a_batch_size.max(b_batch_size);

        let a = self.reshape(vec![
            a_batch_size,
            self.shape()[self.ndim() - 2],
            self.shape()[self.ndim() - 1],
        ])?;

        let b = rhs.reshape(vec![
            b_batch_size,
            rhs.shape()[rhs.ndim() - 2],
            rhs.shape()[rhs.ndim() - 1],
        ])?;

        let y = Tensor::zeros(vec![batch_size, a.shape()[1], b.shape()[2]], self.device);
        for i in 0..batch_size {
            let a = if a_batch_size == 1 {
                &a
            } else {
                &a.get_item(vec![(i, i + 1), (0, a.shape()[1]), (0, a.shape()[2])])?
            };
            let a = a.reshape(vec![a.shape()[1], a.shape()[2]])?;
            let b = if b_batch_size == 1 {
                &b
            } else {
                &b.get_item(vec![(i, i + 1), (0, b.shape()[1]), (0, b.shape()[2])])?
            };
            let b = b.reshape(vec![b.shape()[1], b.shape()[2]])?;
            let c = a.matmul2d(&b)?;
            let c = c.reshape(vec![1, c.shape()[0], c.shape()[1]])?;
            y.set_item(&vec![(i, i + 1), (0, y.shape()[1]), (0, y.shape()[2])], &c)?;
        }

        let mut y_shape = batch_a_shape;
        y_shape.push(a.shape()[1]);
        y_shape.push(b.shape()[2]);
        let y = y.reshape(y_shape)?;
        let op = Op::Matmul(self.clone(), rhs.clone());
        Ok(y.with_op(Some(op)))
    }

    pub fn pow(&self, rhs: &Self) -> Result<Self> {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Pow(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::pow)
    }

    pub fn pow_scalar(&self, rhs: T) -> Result<Self> {
        let scalar = Tensor::from_scalar(rhs, self.device);
        self.pow(&scalar)
    }

    pub fn exp(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Exp(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::exp)
    }

    pub fn ln(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Ln(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::ln)
    }

    pub fn gather(&self, index: &Tensor<B, u32>, axis: usize) -> Result<Self> {
        let input_storage = &*self.storage.borrow();
        let index_storage = &*index.storage.borrow();

        let output_storage = B::gather(
            input_storage,
            index_storage,
            &self.layout,
            &index.layout,
            axis,
        )?;

        let output_shape = index.shape().to_vec();
        let output_stride = Self::compute_stride(&output_shape);
        let layout = Layout::new(output_shape, output_stride, 0);
        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad(),
            Some(Op::Gather(self.clone(), index.clone(), axis)),
        );
        Ok(output)
    }

    pub fn scatter(&self, index: &Tensor<B, u32>, src: &Self, axis: usize) -> Result<()> {
        let input_storage = &mut *self.storage.borrow_mut();
        let index_storage = &*index.storage.borrow();
        let src_storage = &*src.storage.borrow();
        B::scatter(
            input_storage,
            index_storage,
            src_storage,
            &self.layout,
            &index.layout,
            &src.layout,
            axis,
        )
    }

    pub fn scatter_add(&self, index: &Tensor<B, u32>, src: &Self, axis: usize) -> Result<()> {
        let input_storage = &mut *self.storage.borrow_mut();
        let index_storage = &*index.storage.borrow();
        let src_storage = &*src.storage.borrow();
        B::scatter_add(
            input_storage,
            index_storage,
            src_storage,
            &self.layout,
            &index.layout,
            &src.layout,
            axis,
        )
    }

    pub fn index_select(&self, axis: usize, index: &Tensor<B, u32>) -> Result<Self> {
        if axis > self.ndim() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
            });
        }
        let input_storage = &*self.storage.borrow();
        let index_storage = &*index.storage.borrow();

        let mut output_shape = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                output_shape.push(index.len());
            } else {
                output_shape.push(*dim);
            }
        }
        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape, output_stride, 0);

        let output_storage = B::index_select(
            input_storage,
            index_storage,
            &self.layout,
            &index.layout,
            &output_layout,
            axis,
        )?;

        let output = Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad(),
            Some(Op::IndexSelect(self.clone(), index.clone(), axis)),
        );
        Ok(output)
    }

    pub fn index_copy(
        &self,
        axis: usize,
        index: &Tensor<B, u32>,
        src: &Tensor<B, T>,
    ) -> Result<()> {
        self.index_set_impl(axis, index, src, B::index_copy)
    }

    pub fn index_add(&self, axis: usize, index: &Tensor<B, u32>, src: &Tensor<B, T>) -> Result<()> {
        self.index_set_impl(axis, index, src, B::index_add)
    }

    fn index_set_impl<F>(
        &self,
        axis: usize,
        index: &Tensor<B, u32>,
        src: &Tensor<B, T>,
        f: F,
    ) -> Result<()>
    where
        F: for<'a> Fn(
            &'a mut Storage<T>,
            &'a Storage<u32>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
            &'a Layout,
            &'a Layout,
            usize,
        ) -> Result<()>,
    {
        if axis > self.ndim() {
            return Err(Error::ArgumentsError {
                msg: format!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim()),
            });
        }
        let input_storage = &mut *self.storage.borrow_mut();
        let index_storage = &*index.storage.borrow();
        let src_storage = &*src.storage.borrow();

        let mut output_shape = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                output_shape.push(index.len());
            } else {
                output_shape.push(*dim);
            }
        }
        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape, output_stride, 0);

        f(
            input_storage,
            index_storage,
            src_storage,
            &self.layout,
            &index.layout,
            &src.layout,
            &output_layout,
            axis,
        )
    }

    pub fn relu(&self) -> Result<Self> {
        let zero = Tensor::zeros(vec![1], self.device);
        let mask = self.gt(&zero)?.to_dtype::<T>()?;
        self * mask
    }

    pub fn softmax(&self, axis: usize) -> Result<Self> {
        let x_stable = (self - self.max_axis(axis, true)?)?;
        let x_stable_exp = x_stable.exp();
        &x_stable_exp / &x_stable_exp.sum_axis(axis, true)?
    }

    pub fn log_softmax(&self, axis: usize) -> Result<Self> {
        let eps = Tensor::from_scalar(T::from_f64(1e-7), self.device);
        Ok((self.softmax(axis)? + eps)?.ln())
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
                Op::Cat(xs, _, _) => {
                    for x in xs {
                        Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                    }
                }
                Op::GetItem(x, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Contiguous(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Sum(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::SumAxis(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Max(x, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::MaxAxis(x, _) => {
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
                Op::Matmul(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Pow(x1, x2) => {
                    Self::add_work_node(depth + 1, x1.clone(), &mut work_nodes, &mut seen_set);
                    Self::add_work_node(depth + 1, x2.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Exp(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Ln(x) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::Gather(x, _, _) => {
                    Self::add_work_node(depth + 1, x.clone(), &mut work_nodes, &mut seen_set);
                }
                Op::IndexSelect(x, _, _) => {
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

            let gy = grads.get(&node).unwrap().clone();
            match op {
                Op::Reshape(x) => {
                    Self::reshape_backward(&mut grads, &gy, &x)?;
                }
                Op::BroadcastTo(x) => {
                    Self::broadcast_to_backward(&mut grads, &gy, &x)?;
                }
                Op::PermutedAxes(x, axes) => {
                    Self::permuted_axes_backward(&mut grads, &gy, &x, axes)?;
                }
                Op::Cat(xs, axis, split_sections) => {
                    Self::cat_backward(&mut grads, &gy, &xs, axis, &split_sections)?;
                }
                Op::GetItem(x, ranges) => {
                    Self::get_item_backward(&mut grads, &gy, &x, ranges)?;
                }
                Op::Contiguous(x) => {
                    Self::contiguous_backward(&mut grads, &gy, &x)?;
                }
                Op::Sum(x) => {
                    Self::sum_backward(&mut grads, &gy, &x)?;
                }
                Op::SumAxis(x) => {
                    Self::sum_axis_backward(&mut grads, &gy, &x)?;
                }
                Op::Max(x, y) => {
                    Self::max_backward(&mut grads, &gy, &x, &y)?;
                }
                Op::MaxAxis(x, y) => {
                    Self::max_axis_backward(&mut grads, &gy, &x, &y)?;
                }
                Op::Add(x1, x2) => {
                    Self::add_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Sub(x1, x2) => {
                    Self::sub_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Mul(x1, x2) => {
                    Self::mul_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Div(x1, x2) => {
                    Self::div_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Neg(x) => {
                    Self::neg_backward(&mut grads, &gy, &x)?;
                }
                Op::Matmul(x1, x2) => {
                    Self::matmul_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Pow(x1, x2) => {
                    Self::pow_backward(&mut grads, &gy, &x1, &x2)?;
                }
                Op::Exp(x) => {
                    Self::exp_backward(&mut grads, &gy, &x)?;
                }
                Op::Ln(x) => {
                    Self::ln_backward(&mut grads, &gy, &x)?;
                }
                Op::Gather(x, index, axis) => {
                    Self::gather_backward(&mut grads, &gy, &x, &index, axis)?;
                }
                Op::IndexSelect(x, index, axis) => {
                    Self::index_select_backward(&mut grads, &gy, &x, &index, axis)?;
                }
            }
            grads.remove(&node);
        }

        Ok(grads)
    }

    fn reshape_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = gy.reshape(x.shape().to_vec())?;
        grads.add(&x, gx)?;
        Ok(())
    }

    fn broadcast_to_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = gy.sum_to(x.shape())?;
        grads.add(&x, gx)?;
        Ok(())
    }

    fn permuted_axes_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        axes: Vec<usize>,
    ) -> Result<()> {
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
        let gx = gy.permuted_axes(&inv_axes)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn cat_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        xs: &[Tensor<B, T>],
        axis: usize,
        split_sections: &[usize],
    ) -> Result<()> {
        for (i, t) in gy.split(axis, split_sections)?.iter().enumerate() {
            grads.add(&xs[i], t.clone())?;
        }
        Ok(())
    }

    fn get_item_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        ranges: Vec<(usize, usize)>,
    ) -> Result<()> {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device());
        gx.set_item(&ranges, &gy)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn contiguous_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        grads.add(x, gy.clone())?;
        Ok(())
    }

    fn sum_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = gy.broadcast_to(x.shape().to_vec())?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn sum_axis_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = gy.broadcast_to(x.shape().to_vec())?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn max_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        y: &Tensor<B, T>,
    ) -> Result<()> {
        let mask = x.eq(&y)?;
        let gx = (gy * mask.to_dtype::<T>()?)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn max_axis_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        y: &Tensor<B, T>,
    ) -> Result<()> {
        let mask = x.eq(&y)?;
        let gx = (gy * mask.to_dtype::<T>()?)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn add_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<()> {
        let gx1 = gy.sum_to(x1.shape())?;
        let gx2 = gy.sum_to(x2.shape())?;
        grads.add(x1, gx1)?;
        grads.add(x2, gx2)?;
        Ok(())
    }

    fn sub_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<()> {
        let gx1 = gy.sum_to(x1.shape())?;
        let gx2 = -gy.sum_to(x2.shape())?;
        grads.add(x1, gx1)?;
        grads.add(x2, gx2)?;
        Ok(())
    }

    fn mul_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<()> {
        let gx1 = (gy * x2)?.sum_to(x1.shape())?;
        let gx2 = (gy * x1)?.sum_to(x2.shape())?;
        grads.add(x1, gx1)?;
        grads.add(x2, gx2)?;
        Ok(())
    }

    fn div_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<()> {
        let gx1 = (gy / x2)?.sum_to(x1.shape())?;
        let gx2 = (gy * (-x1 / x2.pow_scalar(T::from_f32(2.0))?)?)?.sum_to(x2.shape())?;
        grads.add(x1, gx1)?;
        grads.add(x2, gx2)?;
        Ok(())
    }

    fn matmul_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x0: &Tensor<B, T>,
        x1: &Tensor<B, T>,
    ) -> Result<()> {
        let gx0 = {
            let mut axes = Vec::new();
            for axis in 0..x1.ndim() {
                if axis == x1.ndim() - 1 {
                    axes.push(axis - 1);
                } else if axis == x1.ndim() - 2 {
                    axes.push(axis + 1);
                } else {
                    axes.push(axis);
                }
            }
            let x1_t = x1.permuted_axes(&axes)?;
            let gx0 = gy.matmul(&x1_t)?;
            let gx0 = if gx0.ndim() != x0.ndim() {
                gx0.sum_to(x0.shape())?
            } else {
                gx0
            };
            gx0
        };

        let gx1 = {
            let mut axes = Vec::new();
            for axis in 0..x0.ndim() {
                if axis == x0.ndim() - 1 {
                    axes.push(axis - 1);
                } else if axis == x0.ndim() - 2 {
                    axes.push(axis + 1);
                } else {
                    axes.push(axis);
                }
            }
            let x0_t = x0.permuted_axes(&axes)?;
            let gx1 = x0_t.matmul(gy)?;
            let gx1 = if gx1.ndim() != x1.ndim() {
                gx1.sum_to(x1.shape())?
            } else {
                gx1
            };
            gx1
        };
        grads.add(x0, gx0)?;
        grads.add(x1, gx1)?;
        Ok(())
    }

    fn pow_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) -> Result<()> {
        let one = Tensor::ones(vec![1], gy.device);
        let gx1 = (x2 * x1.pow(&((x2 - one)?))? * gy)?.sum_to(x1.shape())?;
        let gx2 = (gy * x1.ln())?.sum_to(x2.shape())?;
        grads.add(x1, gx1)?;
        grads.add(x2, gx2)?;
        Ok(())
    }

    fn neg_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = -gy;
        grads.add(x, gx)?;
        Ok(())
    }

    fn exp_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
    ) -> Result<()> {
        let gx = (gy * x.exp())?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn ln_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) -> Result<()> {
        let gx = (gy / x)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn gather_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        index: &Tensor<B, u32>,
        axis: usize,
    ) -> Result<()> {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device);
        gx.scatter_add(index, gy, axis)?;
        grads.add(x, gx)?;
        Ok(())
    }

    fn index_select_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        index: &Tensor<B, u32>,
        axis: usize,
    ) -> Result<()> {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device);
        gx.index_add(axis, index, gy)?;
        grads.add(x, gx)?;
        Ok(())
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
            .flat_map(|a| a.to_vec())
            .collect::<Vec<_>>();
        rust_dnn_core::tensor::Tensor::from_vec(data, shape, Device::get_cpu_device()).unwrap()
    }};
    ( $($x:expr),+ $(,)? ) => {{
        let data = vec![$($x),+];
        let shape = vec![data.len()];
        rust_dnn_core::tensor::Tensor::from_vec(data, shape, Device::get_cpu_device()).unwrap()
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
    type Output = Tensor<B, T>;

    fn neg(self) -> Tensor<B, T> {
        self.neg_impl()
    }
}

impl<B: Backend, T: Float> ops::Neg for &Tensor<B, T> {
    type Output = Tensor<B, T>;

    fn neg(self) -> Tensor<B, T> {
        self.neg_impl()
    }
}
