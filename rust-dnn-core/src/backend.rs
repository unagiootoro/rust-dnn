use crate::error::Result;
use crate::float::Float;
use crate::{layout::Layout, num::Num, storage::Storage};

pub trait Backend : Clone + Copy {
    fn storage_to_vec<T: Num>(storage: &Storage, layout: &Layout) -> Vec<T>;
    fn contiguous<T: Num>(storage: &Storage, layout: &Layout) -> Result<Storage>;
    fn sum_axis<T: Num>(
        input_storage: &Storage,
        input_layout: &Layout,
        output_layout: &Layout,
        axis: usize,
    ) -> Result<Storage>;
    fn op_add<T: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage>;
    fn op_sub<T: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage>;
    fn op_mul<T: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage>;
    fn op_div<T: Num>(
        lhs_storage: &Storage,
        rhs_storage: &Storage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Storage>;
    fn op_neg<T: Float>(storage: &Storage, layout: &Layout) -> Result<Storage>;
    fn op_pow_scalar<T: Float>(storage: &Storage, layout: &Layout, rhs: T) -> Result<Storage>;
}
