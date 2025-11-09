use crate::error::Result;
use crate::float::Float;
use crate::{layout::Layout, num::Num, storage::Storage};

pub trait Backend: Clone + Copy {
    fn contiguous<T: Num>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn sum<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>>;
    fn sum_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>>;
    fn max<T: Num>(input_storage: &Storage<T>, input_layout: &Layout) -> Result<Storage<T>>;
    fn max_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>>;
    fn argmax_axis<T: Num>(
        input_storage: &Storage<T>,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<u32>>;
    fn op_add<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn op_add_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()>;
    fn op_sub<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn op_sub_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()>;
    fn op_mul<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn op_mul_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()>;
    fn op_div<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn op_div_assign<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()>;
    fn eq<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>>;
    fn lt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>>;
    fn le<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>>;
    fn gt<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>>;
    fn ge<T: Num>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<u32>>;
    fn op_neg<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn copy<T: Num>(
        lhs_storage: &mut Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<()>;
    fn pow<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn sin<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn cos<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn sqrt<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn exp<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn ln<T: Float>(storage: &Storage<T>, layout: &Layout) -> Result<Storage<T>>;
    fn matmul<T: Float>(
        lhs_storage: &Storage<T>,
        rhs_storage: &Storage<T>,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage<T>>;
    fn gather<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>>;
    fn scatter<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()>;
    fn scatter_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        axis: usize,
    ) -> Result<()>;
    fn index_select<T: Num>(
        input_storage: &Storage<T>,
        index_storage: &Storage<u32>,
        input_layout: &Layout,
        index_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage<T>>;
    fn index_copy<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        dest_shape: &[usize],
        dest_len: usize,
        axis: usize,
    ) -> Result<()>;
    fn index_add<T: Num>(
        input_storage: &mut Storage<T>,
        index_storage: &Storage<u32>,
        src_storage: &Storage<T>,
        input_layout: &Layout,
        index_layout: &Layout,
        src_layout: &Layout,
        dest_shape: &[usize],
        dest_len: usize,
        axis: usize,
    ) -> Result<()>;
}
