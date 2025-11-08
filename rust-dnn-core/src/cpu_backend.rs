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

    fn sum<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        reduce(input_storage, input_layout, |prev, current| prev + current)
    }

    fn sum_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        reduce_axis(
            input_storage,
            input_layout,
            output_layout,
            axis,
            T::zero(),
            |prev, current| prev + current,
        )
    }

    fn max<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>> {
        reduce(input_storage, input_layout, |prev, current| {
            if prev > current { prev } else { current }
        })
    }

    fn max_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        reduce_axis(
            input_storage,
            input_layout,
            output_layout,
            axis,
            T::min_value(),
            |prev, current| if prev > current { prev } else { current },
        )
    }

    fn argmax_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<u32>> {
        reduce_cmp_axis_index(
            input_storage,
            input_layout,
            output_layout,
            axis,
            T::min_value(),
            |prev, current| prev < current,
        )
    }

    fn op_add<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a + b
        })
    }

    fn op_add_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        map_inplace_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            *a += b;
        })
    }

    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a - b
        })
    }

    fn op_sub_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        map_inplace_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            *a -= b;
        })
    }

    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a * b
        })
    }

    fn op_mul_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        map_inplace_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            *a *= b;
        })
    }

    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a / b
        })
    }

    fn op_div_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        map_inplace_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            *a /= b;
        })
    }

    fn eq<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            if a == b { 1 } else { 0 }
        })
    }

    fn lt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            if a < b { 1 } else { 0 }
        })
    }

    fn le<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            if a <= b { 1 } else { 0 }
        })
    }

    fn gt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            if a > b { 1 } else { 0 }
        })
    }

    fn ge<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            if a >= b { 1 } else { 0 }
        })
    }

    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| -a)
    }

    fn copy<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()> {
        map_inplace_arg2::<T, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            *a = b;
        })
    }

    fn pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        map_arg2::<T, _, _>(lhs_storage, rhs_storage, lhs_layout, rhs_layout, |a, b| {
            a.powf(b)
        })
    }

    fn sin<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.sin())
    }

    fn cos<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.cos())
    }

    fn sqrt<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.sqrt())
    }

    fn exp<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.exp())
    }

    fn ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>> {
        map_arg1::<T, _>(storage, layout, |a| a.ln())
    }

    fn matmul<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>> {
        if lhs_layout.ndim() == 2 && rhs_layout.ndim() == 2 {
            matmul2d(lhs_storage, rhs_storage, lhs_layout, rhs_layout)
        } else {
            panic!(
                "Invalid ndim (a.ndim = {}, b.ndim = {})",
                lhs_layout.ndim(),
                rhs_layout.ndim()
            );
        }
    }

    fn gather<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let mut output_data = Vec::new();
        let a_data = input_storage.get_cpu_storage()?;
        let b_data = index_storage.get_cpu_storage()?;

        for i in 0..input_layout.len() {
            let b_index = compute_offset(&index_layout, i);
            let b_value = b_data[b_index];
            let a_index = compute_offset_by_axis_index(
                input_layout.storage_offset(),
                &index_layout.shape(),
                &input_layout.stride(),
                i,
                axis,
                b_value.as_usize(),
            );
            output_data.push(a_data[a_index]);
        }

        let output_storage = Storage::CpuStorage(output_data);
        Ok(output_storage)
    }

    fn scatter<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        scatter_impl(
            input_storage,
            index_storage,
            src_storage,
            input_layout,
            index_layout,
            src_layout,
            axis,
            |a, b| *a = b,
        )
    }

    fn scatter_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        scatter_impl(
            input_storage,
            index_storage,
            src_storage,
            input_layout,
            index_layout,
            src_layout,
            axis,
            |a, b| *a += b,
        )
    }

    fn index_select<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>> {
        let mut output_data = Vec::new();
        let input_data = input_storage.get_cpu_storage()?;
        let index_data = index_storage.get_cpu_storage()?;

        for i in 0..output_layout.len() {
            let output_axis_index = unravel_index_axis(output_layout.shape(), axis, i);

            let index_offset = compute_offset(index_layout, output_axis_index);
            let index_value = index_data[index_offset];

            let input_index = compute_offset_by_axis_index(
                input_layout.storage_offset(),
                &output_layout.shape(),
                &input_layout.stride(),
                i,
                axis,
                index_value as usize,
            );
            output_data.push(input_data[input_index]);
        }

        let output_storage = Storage::CpuStorage(output_data);
        Ok(output_storage)
    }

    fn index_copy<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        index_set_impl(
            input_storage,
            index_storage,
            src_storage,
            input_layout,
            index_layout,
            src_layout,
            output_layout,
            axis,
            |a, b| *a = b,
        )
    }

    fn index_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<()> {
        index_set_impl(
            input_storage,
            index_storage,
            src_storage,
            input_layout,
            index_layout,
            src_layout,
            output_layout,
            axis,
            |a, b| *a += b,
        )
    }
}

fn reduce<T: Num, F>(input_storage: &Storage<T>, input_layout: &Layout, f: F) -> Result<Storage<T>>
where
    F: Fn(T, T) -> T + Sync,
{
    let input_data = input_storage.get_cpu_storage()?;

    let mut result = T::zero();
    for i in 0..input_layout.len() {
        result = f(result, input_data[compute_offset(input_layout, i)]);
    }
    let output_storage = Storage::CpuStorage(vec![result]);
    Ok(output_storage)
}

fn reduce_axis<T: Num, F>(
    input_storage: &Storage<T>,
    input_layout: &Layout,
    output_layout: &Layout,
    axis: usize,
    initial_value: T,
    f: F,
) -> Result<Storage<T>>
where
    F: Fn(T, T) -> T + Sync,
{
    let mut output_data = Vec::new();
    let output_data_len = output_layout.len();
    for _ in 0..output_data_len {
        output_data.push(T::zero());
    }

    let input_data = input_storage.get_cpu_storage()?;

    for i in 0..output_data.len() {
        let mut result = initial_value;
        for j in 0..input_layout.shape()[axis] {
            let input_index = compute_offset_by_axis_index(
                input_layout.storage_offset(),
                output_layout.shape(),
                input_layout.stride(),
                i,
                axis,
                j,
            );
            result = f(result, input_data[input_index]);
        }
        output_data[i] += result;
    }

    let output_storage = Storage::CpuStorage(output_data);
    Ok(output_storage)
}

fn reduce_cmp_axis_index<T: Num, F>(
    input_storage: &Storage<T>,
    input_layout: &Layout,
    output_layout: &Layout,
    axis: usize,
    initial_value: T,
    f: F,
) -> Result<Storage<u32>>
where
    F: Fn(T, T) -> bool,
{
    let mut output_data = Vec::new();
    let output_data_len = output_layout.len();
    for _ in 0..output_data_len {
        output_data.push(0);
    }

    let input_data = input_storage.get_cpu_storage()?;

    for i in 0..output_data.len() {
        let mut result = initial_value;
        let mut result_index = 0u32;
        for j in 0..input_layout.shape()[axis] {
            let input_index = compute_offset_by_axis_index(
                input_layout.storage_offset(),
                output_layout.shape(),
                input_layout.stride(),
                i,
                axis,
                j,
            );
            let updated = f(result, input_data[input_index]);
            if updated {
                result = input_data[input_index];
                result_index = j as u32;
            }
        }
        output_data[i] = result_index;
    }

    let output_storage = Storage::CpuStorage(output_data);
    Ok(output_storage)
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
    Ok(output_storage)
}

fn map_arg2<T: Num, T2: Num, F>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: F,
) -> Result<Storage<T2>>
where
    F: Fn(T, T) -> T2 + Sync,
{
    let a_data = lhs_storage.get_cpu_storage()?;
    let b_data = rhs_storage.get_cpu_storage()?;
    let output_data: Vec<T2> = (0..lhs_layout.len())
        .into_iter()
        .map(|i| {
            let a_index = compute_offset(&lhs_layout, i);
            let b_index = compute_offset(&rhs_layout, i);
            f(a_data[a_index], b_data[b_index])
        })
        .collect();
    let output_storage = Storage::CpuStorage(output_data);
    Ok(output_storage)
}

fn map_inplace_arg2<T: Num, F>(
    lhs_storage: &mut Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
    f: F,
) -> Result<()>
where
    F: Fn(&mut T, T),
{
    let a_data = lhs_storage.get_cpu_storage_mut()?;
    let b_data = rhs_storage.get_cpu_storage()?;
    (0..lhs_layout.len()).into_iter().for_each(|i| {
        let a_index = compute_offset(&lhs_layout, i);
        let b_index = compute_offset(&rhs_layout, i);
        f(&mut a_data[a_index], b_data[b_index]);
    });
    Ok(())
}

fn scatter_impl<T: Num, F>(
    input_storage: &mut Storage<T>,
    index_storage: &Storage<u32>,
    src_storage: &Storage<T>,
    input_layout: &Layout,
    index_layout: &Layout,
    src_layout: &Layout,
    axis: usize,
    f: F,
) -> Result<()>
where
    F: Fn(&mut T, T),
{
    // NOTE: aとbで次元が異なる場合に対応
    let mut b_shape2 = Vec::new();
    if index_layout.ndim() < input_layout.ndim() {
        for _ in 0..(input_layout.ndim() - index_layout.ndim()) {
            b_shape2.push(1);
        }
    }
    for dim in index_layout.shape() {
        b_shape2.push(*dim);
    }

    let a_data = input_storage.get_cpu_storage_mut()?;
    let b_data = index_storage.get_cpu_storage()?;
    let c_data = src_storage.get_cpu_storage()?;

    for i in 0..index_layout.len() {
        let b_index = compute_offset(&index_layout, i);
        let b_value = b_data[b_index];
        // NOTE: indices[axis]の設定で範囲外エラーにならないよう、予めb.shapeはaと同じ次元にしておいたものを使用する。
        let a_index = compute_offset_by_axis_index(
            input_layout.storage_offset(),
            &b_shape2,
            input_layout.stride(),
            i,
            axis,
            b_value.as_usize(),
        );
        let c_index = compute_offset(&src_layout, i);
        f(&mut a_data[a_index], c_data[c_index]);
    }
    Ok(())
}

fn index_set_impl<T: Num, F>(
    input_storage: &mut Storage<T>,
    index_storage: &Storage<u32>,
    src_storage: &Storage<T>,
    input_layout: &Layout,
    index_layout: &Layout,
    src_layout: &Layout,
    output_layout: &Layout,
    axis: usize,
    f: F,
) -> Result<()>
where
    F: Fn(&mut T, T),
{
    let input_data = input_storage.get_cpu_storage_mut()?;
    let index_data = index_storage.get_cpu_storage()?;
    let src_data = src_storage.get_cpu_storage()?;

    for i in 0..output_layout.len() {
        let output_axis_index = unravel_index_axis(output_layout.shape(), axis, i);

        let index_offset = compute_offset(index_layout, output_axis_index);
        let index_value = index_data[index_offset];

        let input_index = compute_offset_by_axis_index(
            input_layout.storage_offset(),
            &output_layout.shape(),
            &input_layout.stride(),
            i,
            axis,
            index_value as usize,
        );

        let src_offset = compute_offset(src_layout, output_axis_index);
        let src_value = src_data[src_offset];
        f(&mut input_data[input_index], src_value);
    }

    Ok(())
}

fn matmul2d<T: Float>(
    lhs_storage: &Storage<T>,
    rhs_storage: &Storage<T>,
    lhs_layout: &Layout,
    rhs_layout: &Layout,
) -> Result<Storage<T>> {
    let a_rows = lhs_layout.shape()[0];
    let a_cols = lhs_layout.shape()[1];
    let b_cols = rhs_layout.shape()[1];

    let mut output_data = vec![T::zero(); b_cols * a_rows];
    let a_data = lhs_storage.get_cpu_storage()?;
    let b_data = rhs_storage.get_cpu_storage()?;

    for i in 0..a_rows {
        for j in 0..b_cols {
            for k in 0..a_cols {
                let result_index = i * b_cols + j;
                let a_index =
                    compute_offset_by_dim2(lhs_layout.storage_offset(), lhs_layout.stride(), i, k);
                let b_index =
                    compute_offset_by_dim2(rhs_layout.storage_offset(), rhs_layout.stride(), k, j);
                output_data[result_index] += a_data[a_index] * b_data[b_index];
            }
        }
    }

    let output_storage = Storage::CpuStorage(output_data);
    Ok(output_storage)
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
    output_shape: &[usize],
    input_stride: &[usize],
    mut linear_index: usize,
    axis: usize,
    index: usize,
) -> usize {
    let mut offset = 0;
    for i in (0..output_shape.len()).rev() {
        if input_stride[i] > 0 {
            let idx = if i == axis {
                index
            } else {
                linear_index % output_shape[i]
            };
            offset += idx * input_stride[i];
        }
        linear_index /= output_shape[i];
    }
    storage_offset + offset
}

#[inline(always)]
fn compute_offset_by_indices(base_offset: usize, strides: &[usize], indices: &[usize]) -> usize {
    let mut offset = 0;
    for i in 0..strides.len() {
        if strides[i] > 0 {
            let idx = indices[i];
            offset += idx * strides[i];
        }
    }
    base_offset + offset
}

#[inline(always)]
fn compute_offset_by_dim2(base_offset: usize, strides: &[usize], i: usize, j: usize) -> usize {
    let mut offset = 0;
    if strides[0] > 0 {
        offset += i * strides[0];
    }
    if strides[1] > 0 {
        offset += j * strides[1];
    }
    base_offset + offset
}

#[inline(always)]
fn unravel_index_axis(shape: &[usize], axis: usize, mut linear_index: usize) -> usize {
    for i in (0..shape.len()).rev() {
        let idx = linear_index % shape[i];
        linear_index /= shape[i];
        if i == axis {
            return idx;
        }
    }
    panic!();
}
