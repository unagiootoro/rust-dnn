use std::{
    cell::RefCell,
    collections::HashSet,
    f64::consts::PI,
    fmt::Arguments,
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
    xorshift128::XorShift128Plus,
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

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>, device: Device<B>) -> Self {
        assert!(shape.len() > 0);

        let len = Self::compute_len(&shape);
        if data.len() != len {
            panic!(
                "Invalid data length(data len = {}, shape len = {})",
                data.len(),
                len
            );
        }
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
        let stride = Self::compute_stride(&shape);
        let storage = match *device.info() {
            DeviceInfo::Cpu => {
                let mut data = Vec::new();
                for _ in 0..Self::compute_len(&shape) {
                    data.push(value);
                }
                Storage::CpuStorage(data)
            }
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => {
                let len = Self::compute_len(&shape);
                match T::dtype() {
                    DType::U32 => {
                        Storage::CudaStorage(GPUBuffer::from_fill_u32(len, value.as_u32()))
                    }
                    DType::F32 => {
                        Storage::CudaStorage(GPUBuffer::from_fill_f32(len, value.as_f32()))
                    }
                    DType::F64 => {
                        Storage::CudaStorage(GPUBuffer::from_fill_f64(len, value.as_f64()))
                    }
                }
            }
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

    pub fn rand_int(
        min: i32,
        max: i32,
        shape: &[usize],
        seed: Option<u64>,
        device: Device<B>,
    ) -> Self {
        let r = Tensor::<B, f32>::rand_uniform(shape, seed, device);
        let r = r * ((max - min) as f64);
        let r = r + min as f64;
        r.to_dtype::<T>()
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

    fn axis_isize_to_usize(axis: isize, ndim: usize) -> Result<usize> {
        if axis >= ndim as isize {
            return Err(Error::ArgumentsError {
                msg: format!("Invalud axis(axis = {}, ndim = {})", axis, ndim),
            });
        }

        if axis >= 0 {
            Ok(axis as usize)
        } else {
            let usize_axis = ((ndim as isize) + axis) as usize;
            if usize_axis >= ndim {
                return Err(Error::ArgumentsError {
                    msg: format!("Invalud axis(axis = {}, ndim = {})", axis, ndim),
                });
            }
            Ok(usize_axis)
        }
    }

    fn validate_axis(axis: isize, ndim: usize) -> Result<()> {
        Self::axis_isize_to_usize(axis, ndim)?;
        Ok(())
    }

    fn unsqueeze_axis_isize_to_usize(axis: isize, ndim: usize) -> Result<usize> {
        if axis > ndim as isize {
            return Err(Error::ArgumentsError {
                msg: format!("Invalud axis(axis = {}, ndim = {})", axis, ndim),
            });
        }

        if axis >= 0 {
            Ok(axis as usize)
        } else {
            let usize_axis = (((ndim + 1) as isize) + axis) as usize;
            if usize_axis > ndim {
                return Err(Error::ArgumentsError {
                    msg: format!("Invalud axis(axis = {}, ndim = {})", axis, ndim),
                });
            }
            Ok(usize_axis)
        }
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

    pub fn size(&self, axis: isize) -> usize {
        let axis = Self::axis_isize_to_usize(axis, self.ndim()).expect("Failed size");
        self.layout.shape()[axis]
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

    pub fn dtype(&self) -> DType {
        T::dtype()
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
            op.is_some(),
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

    fn add_impl(&self, rhs: &Tensor<B, T>) -> Self {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Add(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_add)
    }

    fn add_scalar_impl(&self, rhs: T) -> Self {
        let rhs = Self::from_scalar(rhs, self.device);
        self.add_impl(&rhs)
    }

    pub fn add_assign(&self, rhs: &Tensor<B, T>) {
        self.op2_inplace_impl(rhs, B::op_add_assign)
    }

    fn sub_impl(&self, rhs: &Tensor<B, T>) -> Self {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Sub(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_sub)
    }

    fn sub_scalar_impl(&self, rhs: T) -> Self {
        let rhs = Self::from_scalar(rhs, self.device);
        self.sub_impl(&rhs)
    }

    pub fn sub_assign(&self, rhs: &Tensor<B, T>) {
        self.op2_inplace_impl(rhs, B::op_sub_assign)
    }

    fn mul_impl(&self, rhs: &Tensor<B, T>) -> Self {
        let op: Option<Op<B, T>> = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Mul(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_mul)
    }

    fn mul_scalar_impl(&self, rhs: T) -> Self {
        let rhs = Self::from_scalar(rhs, self.device);
        self.mul_impl(&rhs)
    }

    pub fn mul_assign(&self, rhs: &Tensor<B, T>) {
        self.op2_inplace_impl(rhs, B::op_mul_assign)
    }

    fn div_impl(&self, rhs: &Tensor<B, T>) -> Self {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Div(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::op_div)
    }

    fn div_scalar_impl(&self, rhs: T) -> Self {
        let rhs = Self::from_scalar(rhs, self.device);
        self.div_impl(&rhs)
    }

    pub fn div_assign(&self, rhs: &Tensor<B, T>) {
        self.op2_inplace_impl(rhs, B::op_div_assign)
    }

    pub fn eq(&self, rhs: &Tensor<B, T>) -> Tensor<B, u32> {
        self.op2_impl(rhs, None, B::eq)
    }

    pub fn eq_scalar(&self, rhs: f64) -> Tensor<B, u32> {
        self.op2_impl_scalar(rhs, None, B::eq)
    }

    pub fn lt(&self, rhs: &Tensor<B, T>) -> Tensor<B, u32> {
        self.op2_impl(rhs, None, B::lt)
    }

    pub fn lt_scalar(&self, rhs: f64) -> Tensor<B, u32> {
        self.op2_impl_scalar(rhs, None, B::lt)
    }

    pub fn le(&self, rhs: &Tensor<B, T>) -> Tensor<B, u32> {
        self.op2_impl(rhs, None, B::le)
    }

    pub fn le_scalar(&self, rhs: f64) -> Tensor<B, u32> {
        self.op2_impl_scalar(rhs, None, B::le)
    }

    pub fn gt(&self, rhs: &Tensor<B, T>) -> Tensor<B, u32> {
        self.op2_impl(rhs, None, B::gt)
    }

    pub fn gt_scalar(&self, rhs: f64) -> Tensor<B, u32> {
        self.op2_impl_scalar(rhs, None, B::gt)
    }

    pub fn ge(&self, rhs: &Tensor<B, T>) -> Tensor<B, u32> {
        self.op2_impl(rhs, None, B::ge)
    }

    pub fn ge_scalar(&self, rhs: f64) -> Tensor<B, u32> {
        self.op2_impl_scalar(rhs, None, B::ge)
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

    fn op2_impl<T2: Num, F>(&self, rhs: &Tensor<B, T>, op: Option<Op<B, T2>>, f: F) -> Tensor<B, T2>
    where
        F: for<'a> Fn(
            &'a Storage<T>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
        ) -> Result<Storage<T2>>,
    {
        let broadcasted_shape =
            Self::broadcast_shape(self.shape(), rhs.shape()).expect("Failed op2_impl");
        let lhs = self.detach().broadcast_to(broadcasted_shape.clone());
        let rhs = rhs.detach().broadcast_to(broadcasted_shape.clone());
        let lhs_storage = &*lhs.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();

        let output_storage =
            f(lhs_storage, rhs_storage, &lhs.layout, &rhs.layout).expect("Failed op2_impl");

        let stride = Self::compute_stride(&broadcasted_shape);
        let layout = Layout::new(broadcasted_shape, stride, 0);
        Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            op.is_some(),
            op,
        )
    }

    fn op2_impl_scalar<T2: Num, F>(&self, rhs: f64, op: Option<Op<B, T2>>, f: F) -> Tensor<B, T2>
    where
        F: for<'a> Fn(
            &'a Storage<T>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
        ) -> Result<Storage<T2>>,
    {
        let rhs = Tensor::from_scalar(T::from_f64(rhs), self.device);
        self.op2_impl(&rhs, op, f)
    }

    fn op2_inplace_impl<F>(&self, rhs: &Tensor<B, T>, f: F)
    where
        F: for<'a> Fn(&'a mut Storage<T>, &'a Storage<T>, &'a Layout, &'a Layout) -> Result<()>,
    {
        let broadcasted_shape =
            Self::broadcast_shape(self.shape(), rhs.shape()).expect("Failed op2_inplace_impl");
        if self.shape() != broadcasted_shape {
            panic!(
                "Invalid shape(self.shape = {:?}, broadcasted_shape = {:?})",
                self.shape(),
                broadcasted_shape
            )
        }
        let rhs = rhs.detach().broadcast_to(broadcasted_shape.clone());
        let lhs_storage = &mut *self.storage.borrow_mut();
        let rhs_storage = &*rhs.storage.borrow();

        f(lhs_storage, rhs_storage, &self.layout, &rhs.layout).expect("Failed op2_inplace_impl")
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Self {
        assert!(shape.len() > 0);

        let output_len = Self::compute_len(&shape);
        if self.len() != output_len {
            panic!(
                "Length mismatch (from shape = {:?}, from shape len = {}, to shape = {:?}, to shape len = {})",
                self.shape(),
                self.len(),
                shape,
                output_len
            );
        }
        let input = self.contiguous();
        let stride = Self::compute_stride(&shape);
        let layout = Layout::new(shape, stride, input.storage_offset());
        let op = if self.is_requires_grad {
            Some(Op::Reshape(self.clone()))
        } else {
            None
        };
        return Tensor::new(
            input.storage.clone(),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
    }

    pub fn flatten(&self) -> Self {
        self.reshape(vec![self.len()])
    }

    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::new();
        for dim in self.shape() {
            if *dim != 1 {
                shape.push(*dim);
            }
        }

        if shape.len() == 0 {
            shape.push(1);
        }

        self.reshape(shape)
    }

    pub fn squeeze_axes(&self, axes: &[isize]) -> Self {
        let mut axes2 = Vec::new();
        for axis in axes {
            let axis = Self::axis_isize_to_usize(*axis, self.ndim()).expect("Failed squeeze_axes");
            axes2.push(axis);
        }

        let mut shape = Vec::new();
        for (axis, dim) in self.shape().iter().enumerate() {
            if axes2.contains(&axis) {
                if *dim != 1 {
                    panic!("Dim is not 1(dim = {}, axis = {})", dim, axis);
                }
            } else {
                shape.push(self.shape()[axis]);
            }
        }

        if shape.len() == 0 {
            shape.push(1);
        }

        self.reshape(shape)
    }

    pub fn unsqueeze(&self, axis: isize) -> Self {
        let axis =
            Self::unsqueeze_axis_isize_to_usize(axis, self.ndim()).expect("Failed unsqueeze");
        let mut shape = self.shape().to_vec();
        shape.insert(axis, 1);
        self.reshape(shape)
    }

    pub fn permuted_axes(&self, axes: &[isize]) -> Self {
        if self.ndim() != axes.len() {
            panic!(
                "Mismatch dims(self.ndim() = {}, axes.len() = {})",
                self.ndim(),
                axes.len()
            );
        }

        let mut axes2 = Vec::new();
        for axis in axes {
            let axis = Self::axis_isize_to_usize(*axis, self.ndim()).expect("Failed permuted_axes");
            axes2.push(axis);
        }

        let mut new_shape = Vec::new();
        let mut new_stride = Vec::new();
        for axis in &axes2 {
            new_shape.push(self.shape()[*axis]);
            new_stride.push(self.stride()[*axis]);
        }
        let layout = Layout::new(new_shape, new_stride, self.storage_offset());
        let op = if self.is_requires_grad {
            Some(Op::PermutedAxes(self.clone(), axes2.to_vec()))
        } else {
            None
        };
        return Tensor::new(
            self.storage.clone(),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
    }

    pub fn reversed_axes(&self) -> Self {
        let mut axes = Vec::new();
        for axis in (0..self.ndim()).rev() {
            axes.push(axis as isize);
        }
        self.permuted_axes(&axes)
    }

    pub fn get_item(&self, ranges: Vec<(usize, usize)>) -> Self {
        for (i, range) in ranges.iter().enumerate() {
            if range.1 > self.shape()[i] {
                panic!(
                    "Invalid range(self.shape = {:?}, ranges = {:?})",
                    self.shape(),
                    ranges
                );
            }
        }

        let mut output_shape = Vec::new();
        for range in &ranges {
            output_shape.push(range.1 - range.0);
        }
        let storage_offset =
            Self::compute_offset_by_ranges(self.storage_offset(), self.stride(), &ranges);
        let output_layout = Layout::new(output_shape, self.stride().to_vec(), storage_offset);
        return Tensor::new(
            Rc::clone(&self.storage),
            output_layout,
            self.device,
            self.dtype,
            self.is_requires_grad(),
            Some(Op::GetItem(self.clone(), ranges)),
        );
    }

    pub fn set_item(&self, ranges: &Vec<(usize, usize)>, src: &Self) {
        let x = self.get_item(ranges.clone());
        x.copy(src)
    }

    pub fn select(&self, axis: isize, index: usize) -> Self {
        let axis = Self::axis_isize_to_usize(axis, self.ndim()).expect("Failed select");
        let mut ranges = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                if index >= *dim {
                    panic!(
                        "Invalud index: self.shape = {:?}, axis = {}, index = {}",
                        self.shape(),
                        axis,
                        index,
                    );
                }
                ranges.push((index, index + 1));
            } else {
                ranges.push((0, *dim));
            }
        }
        self.get_item(ranges).squeeze_axes(&[axis as isize])
    }

    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Self {
        if axis >= self.ndim() {
            panic!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim());
        }
        let mut ranges = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                if start + length >= *dim {
                    panic!(
                        "Invalud index: self.shape = {:?}, axis = {}, start = {}, length = {}",
                        self.shape(),
                        axis,
                        start,
                        length,
                    );
                }
                ranges.push((start, start + length));
            } else {
                ranges.push((0, *dim));
            }
        }
        self.get_item(ranges)
    }

    pub fn gather(&self, index: &Tensor<B, u32>, axis: usize) -> Self {
        let input_storage = &*self.storage.borrow();
        let index_storage = &*index.storage.borrow();

        let output_storage = B::gather(
            input_storage,
            index_storage,
            &self.layout,
            &index.layout,
            axis,
        )
        .expect("Failed gather");

        let output_shape = index.shape().to_vec();
        let output_stride = Self::compute_stride(&output_shape);
        let layout = Layout::new(output_shape, output_stride, 0);
        return Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad(),
            Some(Op::Gather(self.clone(), index.clone(), axis)),
        );
    }

    pub fn scatter(&self, index: &Tensor<B, u32>, src: &Self, axis: usize) {
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
        .expect("Failed scatter")
    }

    pub fn scatter_add(&self, index: &Tensor<B, u32>, src: &Self, axis: usize) {
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
        .expect("Failed scatter_add")
    }

    pub fn index_select(&self, axis: usize, index: &Tensor<B, u32>) -> Self {
        if axis > self.ndim() {
            panic!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim());
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
        )
        .expect("Failed index_select");

        Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad(),
            Some(Op::IndexSelect(self.clone(), index.clone(), axis)),
        )
    }

    pub fn index_copy(&self, axis: usize, index: &Tensor<B, u32>, src: &Tensor<B, T>) {
        self.index_set_impl(axis, index, src, B::index_copy)
    }

    pub fn index_add(&self, axis: usize, index: &Tensor<B, u32>, src: &Tensor<B, T>) {
        self.index_set_impl(axis, index, src, B::index_add)
    }

    fn index_set_impl<F>(&self, axis: usize, index: &Tensor<B, u32>, src: &Tensor<B, T>, f: F)
    where
        F: for<'a> Fn(
            &'a mut Storage<T>,
            &'a Storage<u32>,
            &'a Storage<T>,
            &'a Layout,
            &'a Layout,
            &'a Layout,
            &[usize],
            usize,
            usize,
        ) -> Result<()>,
    {
        if axis > self.ndim() {
            panic!("Invalid axis(axis = {}, ndim = {})", axis, self.ndim())
        }
        let input_storage = &mut *self.storage.borrow_mut();
        let index_storage = &*index.storage.borrow();
        let src_storage = &*src.storage.borrow();

        let mut dest_shape = Vec::new();
        for (i, dim) in self.shape().iter().enumerate() {
            if i == axis {
                dest_shape.push(index.len());
            } else {
                dest_shape.push(*dim);
            }
        }

        f(
            input_storage,
            index_storage,
            src_storage,
            &self.layout,
            &index.layout,
            &src.layout,
            &dest_shape,
            Self::compute_len(&dest_shape),
            axis,
        )
        .expect("Failed index_set_impl")
    }

    pub fn cat(tensors: &[Self], axis: isize) -> Self {
        let axis = Self::axis_isize_to_usize(axis, tensors[0].ndim()).expect("cat");

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
            y.set_item(&ranges, &x)
        }

        y.with_op(Some(Op::Cat(tensors.to_vec(), axis, split_sections)))
    }

    pub fn split(&self, axis: usize, split_sections: &[usize]) -> Vec<Self> {
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
            let y = self.get_item(ranges);
            ys.push(y);
        }
        ys
    }

    pub fn repeat_interleave(&self, axis: isize, repeats: usize) -> Self {
        let input_size = self.size(axis) as isize;
        let indices = Tensor::arange(0..input_size, self.device());
        let indices = indices
            .reshape(vec![indices.len(), 1])
            .broadcast_to(vec![indices.len(), repeats])
            .flatten();
        let axis2 = Self::axis_isize_to_usize(axis, self.ndim()).expect("Failed select");
        self.index_select(axis2, &indices)
    }

    pub fn copy(&self, src: &Self) {
        self.op2_inplace_impl(src, B::copy)
    }

    pub fn broadcast_to(&self, shape: Vec<usize>) -> Self {
        let mut input_shape = Vec::new();
        if self.ndim() < shape.len() {
            for _ in 0..(shape.len() - self.ndim()) {
                input_shape.push(1);
            }
        }
        for dim in self.shape() {
            input_shape.push(*dim);
        }
        let input = self.reshape(input_shape.clone());
        let stride = Self::compute_broadcast_stride(input.shape(), input.stride(), &shape)
            .expect("Invalid broadcast_to shape");
        let layout = Layout::new(shape, stride, input.storage_offset());
        let op = if self.is_requires_grad {
            Some(Op::BroadcastTo(self.clone()))
        } else {
            None
        };
        return Tensor::new(
            Rc::clone(&input.storage),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        );
    }

    pub fn masked_fill(&self, mask: &Tensor<B, u32>, value: T) -> Tensor<B, T> {
        let mask_f32 = mask.to_dtype::<f32>();
        let reverse_mask = -mask_f32 + 1.0;
        let reverse_mask = reverse_mask.to_dtype::<T>();
        let a = self * reverse_mask;
        let value = Tensor::from_scalar(value, self.device());
        let mask2 = mask.to_dtype::<T>();
        let b = mask2 * value;
        a + b
    }

    pub fn tril(&self) -> Self {
        let mask = Tensor::zeros(self.shape().to_vec(), self.device);
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        let one = Tensor::ones(vec![1], self.device);
        for i in 0..rows {
            let col_end = (i + 1).min(cols);
            mask.set_item(&vec![(i, i + 1), (0, col_end)], &one);
        }
        self * &mask
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

    pub fn to_dtype<T2: Num>(&self) -> Tensor<B, T2> {
        if T::dtype() == T2::dtype() {
            return unsafe { self.clone().reinterpret_cast_dtype::<T2>() };
        }

        let input_storage = &*self.storage.borrow();
        let output_storage = B::convert_dtype::<T, T2>(input_storage, &self.layout).unwrap();

        let layout = Layout::new(
            self.shape().to_vec(),
            Self::compute_stride(&self.shape()),
            0,
        );
        Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            layout,
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            None,
        )
    }

    pub unsafe fn reinterpret_cast_dtype<T2: Num>(self) -> Tensor<B, T2> {
        unsafe { std::mem::transmute::<Self, Tensor<B, T2>>(self) }
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
    pub fn linspace(start: T, stop: T, num: usize, device: Device<B>) -> Self {
        let mut data = Vec::new();
        for i in 0..num {
            if i == 0 {
                data.push(start);
            } else if i == num - 1 {
                data.push(stop);
            } else {
                data.push(start + ((stop - start) / T::from_usize(num - 1) * T::from_usize(i)));
            }
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

    pub fn rand_norm(shape: &[usize], seed: Option<u64>, device: Device<B>) -> Self {
        let u1 = Self::rand_uniform(shape, seed, device);
        let u2 = Self::rand_uniform(shape, seed, device);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    pub fn rand_uniform(shape: &[usize], seed: Option<u64>, device: Device<B>) -> Self {
        let mut rnd = if let Some(seed) = seed {
            XorShift128Plus::from_seed(seed)
        } else {
            XorShift128Plus::from_seed2()
        };
        let mut data = Vec::new();
        for _ in 0..Self::compute_len(shape) {
            let n = rnd.next_f64();
            data.push(T::from_f64(n));
        }
        let stride = Self::compute_stride(shape);
        let storage = match *device.info() {
            DeviceInfo::Cpu => Storage::CpuStorage(data),
            #[cfg(feature = "cuda")]
            DeviceInfo::Cuda => Storage::CudaStorage(GPUBuffer::from_vec(&data)),
        };
        let layout = Layout::new(shape.to_vec(), stride, 0);
        Self::new(
            Rc::new(RefCell::new(storage)),
            layout,
            device,
            T::dtype(),
            false,
            None,
        )
    }

    fn neg_impl(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Neg(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::op_neg)
    }

    fn matmul2d_basic(&self, rhs: &Self) -> Self {
        let lhs_rows = self.shape()[0];
        let lhs_cols = self.shape()[1];
        let rhs_rows = rhs.shape()[0];
        let rhs_cols = rhs.shape()[1];
        assert_eq!(lhs_cols, rhs_rows, "Incompatible matrix shapes");

        let lhs_storage = &*self.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();
        let output_storage = B::matmul(lhs_storage, &rhs_storage, &self.layout, &rhs.layout)
            .expect("Failed matmul2d_basic");
        let output_shape = vec![lhs_rows, rhs_cols];
        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape, output_stride, 0);
        Self::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            self.device,
            self.dtype,
            false,
            None,
        )
    }

    fn matmul2d_cublas(&self, rhs: &Self) -> Self {
        let lhs = self.contiguous();
        let rhs = rhs.contiguous();

        let lhs_rows = lhs.shape()[0];
        let lhs_cols = lhs.shape()[1];
        let rhs_rows = rhs.shape()[0];
        let rhs_cols = rhs.shape()[1];
        assert_eq!(lhs_cols, rhs_rows, "Incompatible matrix shapes");

        let lhs_storage = &*lhs.storage.borrow();
        let rhs_storage = &*rhs.storage.borrow();
        let output_storage = B::cublas_sgemm(lhs_storage, &rhs_storage, &lhs.layout, &rhs.layout)
            .expect("Failed matmul2d_cublas");
        let output_shape = vec![rhs_cols, lhs_rows];
        let output_stride = Self::compute_stride(&output_shape);
        let output_layout = Layout::new(output_shape, output_stride, 0);
        let output = Self::new(
            Rc::new(RefCell::new(output_storage)),
            output_layout,
            lhs.device,
            lhs.dtype,
            false,
            None,
        );
        output.reversed_axes().contiguous()
    }

    fn matmul2d(&self, rhs: &Self) -> Self {
        // TODO: cublas版は期待値と一致しないので暫定無効
        // if B::is_cublas_supported() && self.dtype() == DType::F32 {
        //     self.matmul2d_cublas(rhs)
        // } else {
        //     self.matmul2d_basic(rhs)
        // }
        self.matmul2d_basic(rhs)
    }

    pub fn matmul(&self, rhs: &Self) -> Self {
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
        ]);

        let b = rhs.reshape(vec![
            b_batch_size,
            rhs.shape()[rhs.ndim() - 2],
            rhs.shape()[rhs.ndim() - 1],
        ]);

        let y = Tensor::zeros(vec![batch_size, a.shape()[1], b.shape()[2]], self.device);
        for i in 0..batch_size {
            let a = if a_batch_size == 1 {
                &a
            } else {
                &a.get_item(vec![(i, i + 1), (0, a.shape()[1]), (0, a.shape()[2])])
            };
            let a = a.reshape(vec![a.shape()[1], a.shape()[2]]);
            let b = if b_batch_size == 1 {
                &b
            } else {
                &b.get_item(vec![(i, i + 1), (0, b.shape()[1]), (0, b.shape()[2])])
            };
            let b = b.reshape(vec![b.shape()[1], b.shape()[2]]);
            let c = a.matmul2d(&b);
            let c = c.reshape(vec![1, c.shape()[0], c.shape()[1]]);
            y.set_item(&vec![(i, i + 1), (0, y.shape()[1]), (0, y.shape()[2])], &c);
        }

        let mut y_shape = batch_a_shape;
        y_shape.push(a.shape()[1]);
        y_shape.push(b.shape()[2]);
        let y = y.reshape(y_shape);
        let op = Op::Matmul(self.clone(), rhs.clone());
        y.with_op(Some(op))
    }

    pub fn sum_to(&self, shape: &[usize]) -> Self {
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
                    panic!(
                        "Invalud sum_to input shape: self.shape = {:?}, target shape = {:?}",
                        self.shape(),
                        shape
                    );
                }
            }

            let mut result = self.clone();
            for i in 0..self.ndim() {
                if self.shape()[i] > 1 && shape2[i] == 1 {
                    result = result.sum_axis(i as isize, true);
                }
            }
            result.reshape(shape.to_vec())
        };
        y
    }

    pub fn sum(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Sum(self.clone()))
        } else {
            None
        };
        self.op_reduce_impl(op, B::sum)
    }

    pub fn sum_axis(&self, axis: isize, keepdims: bool) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::SumAxis(self.clone()))
        } else {
            None
        };
        self.op_reduce_axis_impl(axis, keepdims, op, B::sum_axis)
    }

    pub fn mean(&self) -> Self {
        self.sum() / self.len() as f64
    }

    pub fn mean_axis(&self, axis: isize, keepdims: bool) -> Self {
        self.sum_axis(axis, keepdims) / self.size(axis) as f64
    }

    pub fn mean_axes(&self, axes: &[isize], keepdims: bool) -> Self {
        let mut x = self.clone();
        for axis in axes {
            x = x.mean_axis(*axis, true);
        }
        if !keepdims {
            let mut new_shape = Vec::new();
            for (i, dim) in x.shape().iter().enumerate() {
                if !axes.contains(&(i as isize)) {
                    new_shape.push(*dim);
                }
            }
            x = x.reshape(new_shape);
        }
        x
    }

    pub fn max(&self) -> Self {
        let output = self.op_reduce_impl(None, B::max);
        let op = if self.is_requires_grad() {
            Some(Op::Max(self.clone(), output.detach()))
        } else {
            None
        };
        output.with_op(op)
    }

    pub fn max_axis(&self, axis: isize, keepdims: bool) -> Self {
        let output = self.op_reduce_axis_impl(axis, true, None, B::max_axis);
        let op = if self.is_requires_grad() {
            Some(Op::MaxAxis(self.clone(), output.detach()))
        } else {
            None
        };
        let output = output.with_op(op);
        if keepdims {
            output
        } else {
            output.squeeze_axes(&[axis])
        }
    }

    pub fn argmax_axis(&self, axis: isize, keepdims: bool) -> Tensor<B, u32> {
        self.op_reduce_axis_impl(axis, keepdims, None, B::argmax_axis)
    }

    pub fn cumsum(&self, axis: isize) -> Self {
        Self::validate_axis(axis, self.ndim()).expect("cumsum");

        let mut y_list = Vec::new();
        let dim = self.size(axis);
        let mut t = self.select(axis, 0);
        y_list.push(t.clone());
        for i in 1..dim {
            let v = self.select(axis, i);
            t = t + v;
            y_list.push(t.clone());
        }
        Tensor::cat(&y_list, axis)
    }

    pub fn cumprod(&self, axis: isize) -> Self {
        Self::validate_axis(axis, self.ndim()).expect("cumprod");

        let mut y_list = Vec::new();
        let dim = self.size(axis);
        let mut t = self.select(axis, 0);
        y_list.push(t.clone());
        for i in 1..dim {
            let v = self.select(axis, i);
            t = t * v;
            y_list.push(t.clone());
        }
        Tensor::cat(&y_list, axis)
    }

    pub fn multinomial(&self, num_samples: usize, seed: Option<u64>) -> Tensor<B, u32> {
        let mut probs = self / self.sum_axis(-1, true);
        let mut cdf = probs.cumsum(-1);

        if probs.ndim() == 1 {
            probs = probs.unsqueeze(0);
            cdf = cdf.unsqueeze(0)
        }

        let batch_size = probs.size(0);
        let r = Tensor::rand_uniform(&[batch_size, num_samples], seed, self.device());
        let x = r.unsqueeze(-1).gt(&cdf.unsqueeze(1)).to_dtype::<T>();
        let samples = x.sum_axis(-1, false);
        samples.to_dtype::<u32>().squeeze()
    }

    fn op_reduce_impl<F>(&self, op: Option<Op<B, T>>, f: F) -> Self
    where
        F: for<'a> Fn(&'a Storage<T>, &'a Layout) -> Result<Storage<T>>,
    {
        let storage = &*self.storage.borrow();
        let output_storage = f(storage, &self.layout).expect("Failed op_reduce_impl");
        Tensor::new(
            Rc::new(RefCell::new(output_storage)),
            Layout::new(vec![1], vec![1], 0),
            self.device.clone(),
            self.dtype,
            self.is_requires_grad,
            op,
        )
    }

    fn op_reduce_axis_impl<T2: Num, F>(
        &self,
        axis: isize,
        keepdims: bool,
        op: Option<Op<B, T2>>,
        f: F,
    ) -> Tensor<B, T2>
    where
        F: for<'a> Fn(&'a Storage<T>, &'a Layout, &'a Layout, usize) -> Result<Storage<T2>>,
    {
        let axis =
            Self::axis_isize_to_usize(axis, self.ndim()).expect("Failed op_reduce_axis_impl");

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

        let output_storage =
            f(storage, &self.layout, &output_layout, axis).expect("Failed op_reduce_axis_impl");
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
            output
        } else {
            output.squeeze_axes(&[axis as isize])
        }
    }

    pub fn pow(&self, rhs: &Self) -> Self {
        let op = if self.is_requires_grad() || rhs.is_requires_grad() {
            Some(Op::Pow(self.clone(), rhs.clone()))
        } else {
            None
        };
        self.op2_impl(rhs, op, B::pow)
    }

    pub fn pow_scalar(&self, rhs: f64) -> Self {
        let scalar = Tensor::from_scalar(T::from_f64(rhs), self.device);
        self.pow(&scalar)
    }

    pub fn sin(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Sin(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::sin)
    }

    pub fn cos(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Cos(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::cos)
    }

    pub fn tanh(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Tanh(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::tanh)
    }

    pub fn sqrt(&self) -> Self {
        let op = if self.is_requires_grad() {
            Some(Op::Sqrt(self.clone()))
        } else {
            None
        };
        self.op1_impl(op, B::sqrt)
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

    pub fn sigmoid(&self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    pub fn relu(&self) -> Self {
        let mask = self.gt_scalar(0.0).to_dtype::<T>();
        self * mask
    }

    pub fn leaky_relu(&self, alpha: f64) -> Self {
        let lhs = self.gt_scalar(0.0).to_dtype::<T>();
        let rhs = self.le_scalar(0.0).to_dtype::<T>() * alpha;
        let mask = lhs + rhs;
        self * mask
    }

    pub fn silu(&self) -> Self {
        let sig = 1.0 / (1.0 + (-self).exp());
        self * sig
    }

    pub fn gelu(&self) -> Self {
        0.5 * self * (1.0 + ((2.0 / PI).sqrt() * (self + 0.044715 * self.pow_scalar(3.0))).tanh())
    }

    pub fn dropout(&self, dropout_ratio: f64, is_train: bool, seed: Option<u64>) -> Self {
        if is_train {
            let scale = 1.0 - dropout_ratio;
            let rand = Tensor::<B, T>::rand_uniform(self.shape(), seed, self.device);
            let dropout_ratio = Tensor::from_scalar(T::from_f64(dropout_ratio), self.device);
            let mask = dropout_ratio.ge(&rand).to_dtype::<T>();
            (self * mask) / scale
        } else {
            self.clone()
        }
    }

    pub fn softmax(&self, axis: isize) -> Self {
        let x_stable = self - self.max_axis(axis, true);
        let x_stable_exp = x_stable.exp();
        &x_stable_exp / &x_stable_exp.sum_axis(axis, true)
    }

    pub fn log_softmax(&self, axis: isize) -> Self {
        (self.softmax(axis) + 1e-7).ln()
    }

    pub fn im2col(
        &self,
        out_h: usize,
        out_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Self {
        let bsize = self.shape()[0];
        let ch = self.shape()[1];
        let img_h = self.shape()[2];
        let img_w = self.shape()[3];

        let input = self.contiguous();
        let col = Tensor::zeros(vec![bsize, ch * fil_h * fil_w, out_h * out_w], self.device);
        B::im2col(
            &*input.storage.borrow(),
            &mut *col.storage.borrow_mut(),
            ch,
            img_h,
            img_w,
            out_h,
            out_w,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
        )
        .expect("Failed im2col");

        col.with_op(Some(Op::Im2col {
            x: self.clone(),
            out_h,
            out_w,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
        }))
    }

    pub fn col2im(
        &self,
        img_shape: Vec<usize>,
        in_h: usize,
        in_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Self {
        let bsize = img_shape[0];
        let ch = img_shape[1];
        let img_h = img_shape[2];
        let img_w = img_shape[3];

        let input = self.contiguous();
        let img = Tensor::zeros(img_shape.clone(), self.device);
        B::col2im(
            &*input.storage.borrow(),
            &mut *img.storage.borrow_mut(),
            &img_shape,
            bsize,
            ch,
            img_h,
            img_w,
            in_h,
            in_w,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
        )
        .expect("Failed col2im");

        let img = img.with_op(Some(Op::Col2im {
            x: self.clone(),
            in_h,
            in_w,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
        }));

        img
    }

    pub fn zero_padding2d(&self, pad_h: usize, pad_w: usize) -> Self {
        let bsize = self.shape()[0];
        let ch = self.shape()[1];
        let img_h = self.shape()[2];
        let img_w = self.shape()[3];
        let output = Tensor::zeros(vec![bsize, ch, img_h + pad_h, img_w + pad_w], self.device);
        let i_begin = pad_h / 2;
        let i_end = i_begin + img_h;
        let j_begin = pad_w / 2;
        let j_end = j_begin + img_w;
        output.set_item(
            &vec![(0, bsize), (0, ch), (i_begin, i_end), (j_begin, j_end)],
            &self,
        );
        output.with_op(Some(Op::ZeroPadding2d(self.clone(), pad_h, pad_w)))
    }

    pub fn cropping2d(&self, pad_h: usize, pad_w: usize) -> Self {
        let bsize = self.shape()[0];
        let ch = self.shape()[1];
        let img_h = self.shape()[2];
        let img_w = self.shape()[3];
        let i_begin = pad_h / 2;
        let i_end = img_h - (pad_h as f64 / 2.0).round() as usize;
        let j_begin = pad_w / 2;
        let j_end = img_w - (pad_w as f64 / 2.0).round() as usize;
        self.get_item(vec![
            (0, bsize),
            (0, ch),
            (i_begin, i_end),
            (j_begin, j_end),
        ])
    }

    pub fn conv2d(
        &self,
        w: &Self,
        b: Option<&Self>,
        in_filters: usize,
        out_filters: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Option<(usize, usize)>,
        auto_padding: bool,
    ) -> Self {
        let prev_h = self.shape()[2];
        let prev_w = self.shape()[3];

        let (pad_h, pad_w) = if let Some(padding) = &padding {
            (padding.0, padding.1)
        } else {
            if auto_padding {
                Self::compute_conv2d_padding_size(prev_h, prev_w, fil_h, fil_w, stride_h, stride_w)
            } else {
                (0, 0)
            }
        };

        let x = if pad_h > 0 || pad_w > 0 {
            self.zero_padding2d(pad_h, pad_w)
        } else {
            self.clone()
        };

        let (out_h, out_w) = Self::compute_conv2d_out_size(
            prev_h, prev_w, fil_h, fil_w, pad_h, pad_w, stride_h, stride_w,
        );

        let col = x.im2col(out_h, out_w, fil_h, fil_w, stride_h, stride_w);
        let col = col.permuted_axes(&[0, 2, 1]);
        let col = col.reshape(vec![col.shape()[0] * col.shape()[1], col.shape()[2]]);

        let w = w.reshape(vec![
            w.shape()[0],
            w.shape()[1] * w.shape()[2] * w.shape()[3],
        ]);
        let mut y = col.matmul(&w.reversed_axes());
        if let Some(b) = b {
            y = y + b;
        }

        let target_shape = vec![x.shape()[0], y.shape()[1], out_h, out_w];
        let y = y
            .reshape(vec![x.shape()[0], out_h * out_w, y.shape()[1]])
            .permuted_axes(&[0, 2, 1])
            .reshape(target_shape);
        y
    }

    fn compute_conv2d_out_size(
        prev_h: usize,
        prev_w: usize,
        fil_h: usize,
        fil_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> (usize, usize) {
        let out_h = (prev_h + pad_h - fil_h) / stride_h + 1;
        let out_w = (prev_w + pad_w - fil_w) / stride_w + 1;
        (out_h, out_w)
    }

    fn compute_conv2d_padding_size(
        prev_h: usize,
        prev_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> (usize, usize) {
        let out_h = prev_h / stride_h;
        let out_w = prev_w / stride_w;
        let pad_h = out_h * stride_h - prev_h + fil_h - stride_h;
        let pad_w = out_w * stride_w - prev_w + fil_w - stride_w;
        (pad_h, pad_w)
    }

    pub fn deconv2d(
        &self,
        w: &Self,
        b: Option<&Self>,
        in_filters: usize,
        out_filters: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Option<(usize, usize)>,
        auto_padding: bool,
    ) -> Self {
        let bsize = self.shape()[0];
        let prev_h = self.shape()[2];
        let prev_w = self.shape()[3];

        let (pad_h, pad_w) = if let Some(padding) = &padding {
            (padding.0, padding.1)
        } else {
            if auto_padding {
                Self::compute_deconv2d_padding_size(
                    prev_h, prev_w, fil_h, fil_w, stride_h, stride_w,
                )
            } else {
                (0, 0)
            }
        };

        let (out_h, out_w) = Self::compute_deconv2d_out_size(
            prev_h, prev_w, fil_h, fil_w, pad_h, pad_w, stride_h, stride_w,
        );

        let target_shape = vec![self.shape()[0], prev_h, prev_w, self.shape()[1]];
        let x = self
            .reshape(vec![self.shape()[0], self.shape()[1], prev_h * prev_w])
            .permuted_axes(&[0, 2, 1])
            .reshape(target_shape);

        let x = x.reshape(vec![bsize * prev_h * prev_w, in_filters]);
        let w = w.reshape(vec![
            w.shape()[0],
            w.shape()[1] * w.shape()[2] * w.shape()[3],
        ]);
        let col = x.matmul(&w);
        let col = col.reshape(vec![bsize, prev_h * prev_w, out_filters * fil_h * fil_w]);
        let col = col.permuted_axes(&[0, 2, 1]);
        let img_shape = vec![bsize, out_filters, out_h + pad_h, out_w + pad_w];

        let mut y = col.col2im(img_shape, prev_h, prev_w, fil_h, fil_w, stride_h, stride_w);
        if let Some(b) = b {
            y = y + b.reshape(vec![1, b.shape()[0], 1, 1]);
        }

        let output = if pad_h > 0 || pad_w > 0 {
            y.cropping2d(pad_h, pad_w)
        } else {
            y
        };
        output
    }

    fn compute_deconv2d_out_size(
        prev_h: usize,
        prev_w: usize,
        fil_h: usize,
        fil_w: usize,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> (usize, usize) {
        let out_h = (prev_h - 1) * stride_h + fil_h - pad_h;
        let out_w = (prev_w - 1) * stride_w + fil_w - pad_w;
        (out_h, out_w)
    }

    fn compute_deconv2d_padding_size(
        prev_h: usize,
        prev_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> (usize, usize) {
        let out_h = prev_h * stride_h;
        let out_w = prev_w * stride_w;
        let pad_h = (prev_h - 1) * stride_h + fil_h - out_h;
        let pad_w = (prev_w - 1) * stride_w + fil_w - out_w;
        (pad_h, pad_w)
    }

    pub fn max_pool2d(
        &self,
        pool_h: usize,
        pool_w: usize,
        strides: Option<(usize, usize)>,
        padding: Option<(usize, usize)>,
    ) -> Self {
        let x = if let Some((pad_h, pad_w)) = padding {
            self.zero_padding2d(pad_h, pad_w)
        } else {
            self.clone()
        };
        let (stride_h, stride_w) = if let Some(strides) = strides {
            strides
        } else {
            (pool_h, pool_w)
        };

        let input_shape = x.shape();
        let batch_size = input_shape[0];
        let ch = input_shape[1];
        let prev_h = input_shape[2];
        let prev_w = input_shape[3];

        let (pad_h, pad_w) = if let Some(padding) = padding {
            (padding.0, padding.1)
        } else {
            (0, 0)
        };

        let (out_h, out_w) = Self::compute_conv2d_out_size(
            prev_h, prev_w, pool_h, pool_w, pad_h, pad_w, stride_h, stride_w,
        );

        let x = x.im2col(out_h, out_w, pool_h, pool_w, stride_h, stride_w);
        let x = x.reshape(vec![batch_size, ch, pool_h * pool_w, out_h * out_w]);
        x.max_axis(2, false)
            .reshape(vec![batch_size, ch, out_h, out_w])
    }
}

impl<B: Backend, T: Float> Tensor<B, T> {
    pub fn sorted_nodes(&self) -> Vec<Tensor<B, T>> {
        let mut nodes: Vec<Tensor<B, T>> = Vec::new();
        let mut visiting: HashSet<usize> = HashSet::new();
        let mut visited: HashSet<usize> = HashSet::new();
        Self::visit(self, &mut nodes, &mut visiting, &mut visited);
        nodes.reverse();
        nodes
    }

    fn visit(
        node: &Tensor<B, T>,
        sorted: &mut Vec<Tensor<B, T>>,
        visiting: &mut HashSet<usize>,
        visited: &mut HashSet<usize>,
    ) {
        let Some(op) = node.op.clone() else {
            return;
        };

        if visited.contains(&node.id) {
            return;
        }

        if visiting.contains(&node.id) {
            panic!("Tensor graph loop detected. (node.id = {})", node.id);
        }

        visiting.insert(node.id);

        for input in Self::get_op_inputs(op) {
            Self::visit(&input, sorted, visiting, visited);
        }

        visiting.remove(&node.id);
        visited.insert(node.id);
        sorted.push(node.clone());
    }

    fn get_op_inputs(op: Op<B, T>) -> Vec<Tensor<B, T>> {
        match op {
            Op::Reshape(x) => {
                vec![x]
            }
            Op::BroadcastTo(x) => {
                vec![x]
            }
            Op::PermutedAxes(x, _) => {
                vec![x]
            }
            Op::Cat(xs, _, _) => xs,
            Op::GetItem(x, _) => {
                vec![x]
            }
            Op::Contiguous(x) => {
                vec![x]
            }
            Op::Sum(x) => {
                vec![x]
            }
            Op::SumAxis(x) => {
                vec![x]
            }
            Op::Max(x, _) => {
                vec![x]
            }
            Op::MaxAxis(x, _) => {
                vec![x]
            }
            Op::Add(x1, x2) => {
                vec![x1, x2]
            }
            Op::Sub(x1, x2) => {
                vec![x1, x2]
            }
            Op::Mul(x1, x2) => {
                vec![x1, x2]
            }
            Op::Div(x1, x2) => {
                vec![x1, x2]
            }
            Op::Neg(x) => {
                vec![x]
            }
            Op::Matmul(x1, x2) => {
                vec![x1, x2]
            }
            Op::Pow(x1, x2) => {
                vec![x1, x2]
            }
            Op::Sin(x) => {
                vec![x]
            }
            Op::Cos(x) => {
                vec![x]
            }
            Op::Tanh(x) => {
                vec![x]
            }
            Op::Sqrt(x) => {
                vec![x]
            }
            Op::Exp(x) => {
                vec![x]
            }
            Op::Ln(x) => {
                vec![x]
            }
            Op::Gather(x, _, _) => {
                vec![x]
            }
            Op::IndexSelect(x, _, _) => {
                vec![x]
            }
            Op::Im2col { x, .. } => {
                vec![x]
            }
            Op::Col2im { x, .. } => {
                vec![x]
            }
            Op::ZeroPadding2d(x, _, _) => {
                vec![x]
            }
        }
    }

    pub fn backward(&self) -> Gradients<B, T> {
        let mut grads = Gradients::new();
        if !self.is_requires_grad {
            return grads;
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
                    Self::reshape_backward(&mut grads, &gy, &x);
                }
                Op::BroadcastTo(x) => {
                    Self::broadcast_to_backward(&mut grads, &gy, &x);
                }
                Op::PermutedAxes(x, axes) => {
                    Self::permuted_axes_backward(&mut grads, &gy, &x, axes);
                }
                Op::Cat(xs, axis, split_sections) => {
                    Self::cat_backward(&mut grads, &gy, &xs, axis, &split_sections);
                }
                Op::GetItem(x, ranges) => {
                    Self::get_item_backward(&mut grads, &gy, &x, ranges);
                }
                Op::Contiguous(x) => {
                    Self::contiguous_backward(&mut grads, &gy, &x);
                }
                Op::Sum(x) => {
                    Self::sum_backward(&mut grads, &gy, &x);
                }
                Op::SumAxis(x) => {
                    Self::sum_axis_backward(&mut grads, &gy, &x);
                }
                Op::Max(x, y) => {
                    Self::max_backward(&mut grads, &gy, &x, &y);
                }
                Op::MaxAxis(x, y) => {
                    Self::max_axis_backward(&mut grads, &gy, &x, &y);
                }
                Op::Add(x1, x2) => {
                    Self::add_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Sub(x1, x2) => {
                    Self::sub_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Mul(x1, x2) => {
                    Self::mul_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Div(x1, x2) => {
                    Self::div_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Neg(x) => {
                    Self::neg_backward(&mut grads, &gy, &x);
                }
                Op::Matmul(x1, x2) => {
                    Self::matmul_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Pow(x1, x2) => {
                    Self::pow_backward(&mut grads, &gy, &x1, &x2);
                }
                Op::Sin(x) => {
                    Self::sin_backward(&mut grads, &gy, &x);
                }
                Op::Cos(x) => {
                    Self::cos_backward(&mut grads, &gy, &x);
                }
                Op::Tanh(x) => {
                    Self::tanh_backward(&mut grads, &gy, &x);
                }
                Op::Sqrt(x) => {
                    Self::sqrt_backward(&mut grads, &gy, &x);
                }
                Op::Exp(x) => {
                    Self::exp_backward(&mut grads, &gy, &x);
                }
                Op::Ln(x) => {
                    Self::ln_backward(&mut grads, &gy, &x);
                }
                Op::Gather(x, index, axis) => {
                    Self::gather_backward(&mut grads, &gy, &x, &index, axis);
                }
                Op::IndexSelect(x, index, axis) => {
                    Self::index_select_backward(&mut grads, &gy, &x, &index, axis);
                }
                Op::Im2col {
                    x,
                    out_h,
                    out_w,
                    fil_h,
                    fil_w,
                    stride_h,
                    stride_w,
                } => {
                    Self::im2col_backward(
                        &mut grads, &gy, &x, out_h, out_w, fil_h, fil_w, stride_h, stride_w,
                    );
                }
                Op::Col2im {
                    x,
                    in_h,
                    in_w,
                    fil_h,
                    fil_w,
                    stride_h,
                    stride_w,
                } => {
                    Self::col2im_backward(
                        &mut grads, &gy, &x, in_h, in_w, fil_h, fil_w, stride_h, stride_w,
                    );
                }
                Op::ZeroPadding2d(x, pad_h, pad_w) => {
                    Self::zero_padding2d_backward(&mut grads, &gy, &x, pad_h, pad_w);
                }
            }
            grads.remove(&node);
        }

        grads
    }

    fn reshape_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy.reshape(x.shape().to_vec());
        grads.add(&x, gx);
    }

    fn broadcast_to_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy.sum_to(x.shape());
        grads.add(&x, gx);
    }

    fn permuted_axes_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        axes: Vec<usize>,
    ) {
        if gy.ndim() != axes.len() {
            panic!(
                "Mismatch dims(gy.ndim() = {}, self.axes.len() = {})",
                gy.ndim(),
                axes.len()
            );
        }
        let mut inv_axes = Vec::new();
        for i in 0..gy.ndim() {
            let position = axes.iter().position(|&axis| axis == i).unwrap();
            inv_axes.push(position as isize);
        }
        let gx = gy.permuted_axes(&inv_axes);
        grads.add(x, gx);
    }

    fn cat_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        xs: &[Tensor<B, T>],
        axis: usize,
        split_sections: &[usize],
    ) {
        for (i, t) in gy.split(axis, split_sections).iter().enumerate() {
            grads.add(&xs[i], t.clone());
        }
    }

    fn get_item_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        ranges: Vec<(usize, usize)>,
    ) {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device());
        gx.set_item(&ranges, &gy);
        grads.add(x, gx);
    }

    fn contiguous_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        grads.add(x, gy.clone());
    }

    fn sum_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy.broadcast_to(x.shape().to_vec());
        grads.add(x, gx);
    }

    fn sum_axis_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy.broadcast_to(x.shape().to_vec());
        grads.add(x, gx);
    }

    fn max_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        y: &Tensor<B, T>,
    ) {
        let mask = x.eq(&y);
        let gx = gy * mask.to_dtype::<T>();
        grads.add(x, gx);
    }

    fn max_axis_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        y: &Tensor<B, T>,
    ) {
        let mask = x.eq(&y);
        let gx = gy * mask.to_dtype::<T>();
        grads.add(x, gx);
    }

    fn add_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) {
        let gx1 = gy.sum_to(x1.shape());
        let gx2 = gy.sum_to(x2.shape());
        grads.add(x1, gx1);
        grads.add(x2, gx2);
    }

    fn sub_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) {
        let gx1 = gy.sum_to(x1.shape());
        let gx2 = -gy.sum_to(x2.shape());
        grads.add(x1, gx1);
        grads.add(x2, gx2);
    }

    fn mul_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) {
        let gx1 = (gy * x2).sum_to(x1.shape());
        let gx2 = (gy * x1).sum_to(x2.shape());
        grads.add(x1, gx1);
        grads.add(x2, gx2);
    }

    fn div_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) {
        let gx1 = (gy / x2).sum_to(x1.shape());
        let gx2 = (gy * (-x1 / x2.pow_scalar(2.0))).sum_to(x2.shape());
        grads.add(x1, gx1);
        grads.add(x2, gx2);
    }

    fn matmul_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x0: &Tensor<B, T>,
        x1: &Tensor<B, T>,
    ) {
        let gx0 = {
            let mut axes = Vec::new();
            for axis in 0..x1.ndim() {
                if axis == x1.ndim() - 1 {
                    axes.push(axis as isize - 1);
                } else if axis == x1.ndim() - 2 {
                    axes.push(axis as isize + 1);
                } else {
                    axes.push(axis as isize);
                }
            }
            let x1_t = x1.permuted_axes(&axes);
            let gx0 = gy.matmul(&x1_t);
            let gx0 = if gx0.ndim() != x0.ndim() {
                gx0.sum_to(x0.shape())
            } else {
                gx0
            };
            gx0
        };

        let gx1 = {
            let mut axes = Vec::new();
            for axis in 0..x0.ndim() {
                if axis == x0.ndim() - 1 {
                    axes.push(axis as isize - 1);
                } else if axis == x0.ndim() - 2 {
                    axes.push(axis as isize + 1);
                } else {
                    axes.push(axis as isize);
                }
            }
            let x0_t = x0.permuted_axes(&axes);
            let gx1 = x0_t.matmul(gy);
            let gx1 = if gx1.ndim() != x1.ndim() {
                gx1.sum_to(x1.shape())
            } else {
                gx1
            };
            gx1
        };
        grads.add(x0, gx0);
        grads.add(x1, gx1);
    }

    fn pow_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x1: &Tensor<B, T>,
        x2: &Tensor<B, T>,
    ) {
        let gx1 = (x2 * x1.pow(&(x2 - 1.0)) * gy).sum_to(x1.shape());
        let gx2 = (gy * x1.ln()).sum_to(x2.shape());
        grads.add(x1, gx1);
        grads.add(x2, gx2);
    }

    fn neg_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = -gy;
        grads.add(x, gx);
    }

    fn sin_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy * x.cos();
        grads.add(x, gx);
    }

    fn cos_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy * -x.sin();
        grads.add(x, gx);
    }

    fn tanh_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let y = x.tanh();
        let gx = gy * (1.0 - y.pow_scalar(2.0));
        grads.add(x, gx);
    }

    fn sqrt_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy * (1.0 / (2.0 * x.sqrt()));
        grads.add(x, gx);
    }

    fn exp_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy * x.exp();
        grads.add(x, gx);
    }

    fn ln_backward(grads: &mut Gradients<B, T>, gy: &Tensor<B, T>, x: &Tensor<B, T>) {
        let gx = gy / x;
        grads.add(x, gx);
    }

    fn gather_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        index: &Tensor<B, u32>,
        axis: usize,
    ) {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device);
        gx.scatter_add(index, gy, axis);
        grads.add(x, gx);
    }

    fn index_select_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        index: &Tensor<B, u32>,
        axis: usize,
    ) {
        let gx = Tensor::zeros(x.shape().to_vec(), gy.device);
        gx.index_add(axis, index, gy);
        grads.add(x, gx);
    }

    fn im2col_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        out_h: usize,
        out_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) {
        let gx = gy.col2im(
            x.shape().to_vec(),
            out_h,
            out_w,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
        );
        grads.add(x, gx);
    }

    fn col2im_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        in_h: usize,
        in_w: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) {
        let gx = gy.im2col(in_h, in_w, fil_h, fil_w, stride_h, stride_w);
        grads.add(x, gx);
    }

    fn zero_padding2d_backward(
        grads: &mut Gradients<B, T>,
        gy: &Tensor<B, T>,
        x: &Tensor<B, T>,
        pad_h: usize,
        pad_w: usize,
    ) {
        let gx = gy.cropping2d(pad_h, pad_w);
        grads.add(x, gx);
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
        rust_dnn_core::tensor::Tensor::from_vec(data, shape, Device::get_cpu_device())
    }};
    ( $($x:expr),+ $(,)? ) => {{
        let data = vec![$($x),+];
        let shape = vec![data.len()];
        rust_dnn_core::tensor::Tensor::from_vec(data, shape, Device::get_cpu_device())
    }};
}

macro_rules! define_op_arg2 {
    ($op_name:ident, $fn_name:ident, $impl_fn_name:ident, $scalar_impl_fn_name:ident) => {
        impl<B: Backend, T: Num> std::ops::$op_name<Tensor<B, T>> for Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Tensor<B, T> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<&Tensor<B, T>> for Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Tensor<B, T> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<Tensor<B, T>> for &Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Tensor<B, T> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<&Tensor<B, T>> for &Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Tensor<B, T> {
                self.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<f64> for Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: f64) -> Tensor<B, T> {
                self.$scalar_impl_fn_name(T::from_f64(rhs))
            }
        }

        impl<B: Backend, T: Num> std::ops::$op_name<f64> for &Tensor<B, T> {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: f64) -> Tensor<B, T> {
                self.$scalar_impl_fn_name(T::from_f64(rhs))
            }
        }

        impl<B: Backend, T: Float> std::ops::$op_name<Tensor<B, T>> for f64 {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: Tensor<B, T>) -> Tensor<B, T> {
                let lhs = Tensor::<B, T>::from_scalar(T::from_f64(self), rhs.device());
                lhs.$impl_fn_name(&rhs)
            }
        }

        impl<B: Backend, T: Float> std::ops::$op_name<&Tensor<B, T>> for f64 {
            type Output = Tensor<B, T>;

            fn $fn_name(self, rhs: &Tensor<B, T>) -> Tensor<B, T> {
                let lhs = Tensor::<B, T>::from_scalar(T::from_f64(self), rhs.device());
                lhs.$impl_fn_name(&rhs)
            }
        }
    };
}

define_op_arg2!(Add, add, add_impl, add_scalar_impl);
define_op_arg2!(Sub, sub, sub_impl, sub_scalar_impl);
define_op_arg2!(Mul, mul, mul_impl, mul_scalar_impl);
define_op_arg2!(Div, div, div_impl, div_scalar_impl);

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
