mod test_utils;
use rust_dnn_core::{device::Device, error::Result, tensor::Tensor};

use crate::test_utils::assert_tensor;

#[test]
fn test_add_u32() -> Result<()> {
    let device = Device::get_cpu_device();
    let x1 = Tensor::from_vec(vec![1, 2, 3, 4, 5, 6], vec![2, 3], device)?;
    let x2 = Tensor::from_vec(vec![7, 8, 9, 10, 11, 12], vec![2, 3], device)?;
    let y = (x1 + x2)?;
    assert_tensor(
        &y,
        &Tensor::from_vec(vec![8, 10, 12, 14, 16, 18], vec![2, 3], device)?,
    );
    Ok(())
}
