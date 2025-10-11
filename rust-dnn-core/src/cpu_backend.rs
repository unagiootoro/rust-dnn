use crate::backend::Backend;
use crate::error::Result;
use crate::float::Float;
use crate::num::Num;
use crate::{layout::Layout, storage::Storage};

#[derive(Debug, Clone, Copy)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn contiguous<T: Num>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        let mut output_data = Vec::with_capacity(layout.len());
        let input_data = storage.get_cpu_storage()?;
        for i in 0..layout.len() {
            let offset = compute_offset(&layout, i);
            output_data.push(input_data[offset]);
        }

        let output_storage = Storage::CpuStorage(output_data);
        Ok(output_storage)
    }

    fn sum_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let mut output_data = Vec::new();
        let output_data_len = output_layout.len();
        for _ in 0..output_data_len {
            output_data.push(T::zero());
        }

        let input_data = input_storage.get_cpu_storage()?;

        for i in 0..output_data.len() {
            let mut sum = T::zero();
            for j in 0..input_layout.shape()[axis] {
                let input_index = compute_offset_by_axis_index(
                    input_layout.storage_offset(),
                    output_layout.shape(),
                    input_layout.stride(),
                    i,
                    axis,
                    j,
                );
                sum += input_data[input_index];
            }
            output_data[i] += sum;
        }

        let output_storage = Storage::CpuStorage(output_data);
        Ok(output_storage)
    }

    fn op_add<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a + b
        })
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a - b
        })
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a * b
        })
    }

    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a / b
        })
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| -a)
    }

    fn op_pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a.powf(b)
        })
    }

    fn op_ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.ln())
    }
}

fn map_arg1<T: Num, F>(storage: &Storage<T>, layout: &Layout, f: F) -> Result<Storage<T>>
where
    F: Fn(T) -> T + Sync,
{
    let input_data = storage.get_cpu_storage()?;
    let output_data: Vec<T> = (0..layout.len())
        .into_iter()
        .map(|i| {
            let input_index = compute_offset(&layout, i);
            f(input_data[input_index])
        })
        .collect();
    let output_storage = Storage::CpuStorage(output_data);
    Result::Ok(output_storage)
}

fn map_arg2<T: Num, F>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: F,
) -> Result<Storage<T>>
where
    F: Fn(T, T) -> T + Sync,
{
    let a_data = lhs_storage.get_cpu_storage()?;
    let b_data = rhs_storage.get_cpu_storage()?;
    let output_data: Vec<T> = (0..lhs_layout.len())
        .into_iter()
        .map(|i| {
            let a_index = compute_offset(&lhs_layout, i);
            let b_index = compute_offset(&rhs_layout, i);
            f(a_data[a_index], b_data[b_index])
        })
        .collect();
    let output_storage = Storage::CpuStorage(output_data);
    Result::Ok(output_storage)
}

fn map_arg2_scalar<T: Num, F>(
    storage: &Storage<T>,
    layout: &Layout,
    rhs: T,
    f: F,
) -> Result<Storage<T>>
where
    F: Fn(T, T) -> T + Sync,
{
    let input_data = storage.get_cpu_storage()?;
    let output_data: Vec<T> = (0..layout.len())
        .into_iter()
        .map(|i| {
            let input_index = compute_offset(&layout, i);
            f(input_data[input_index], rhs)
        })
        .collect();
    let output_storage = Storage::CpuStorage(output_data);
    Result::Ok(output_storage)
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

fn compute_offset_by_axis_index(
    storage_offset: usize,
    shape: &[usize],
    stride: &[usize],
    mut linear_index: usize,
    axis: usize,
    index: usize,
) -> usize {
    let mut offset = 0;
    for i in (0..shape.len()).rev() {
        if stride[i] > 0 {
            let idx = if i == axis {
                index
            } else {
                linear_index % shape[i]
            };
            offset += idx * stride[i];
        }
        linear_index /= shape[i];
    }
    storage_offset + offset
}
