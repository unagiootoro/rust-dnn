mod test_utils;
use rust_dnn_core::{device::Device, error::Result, ten, tensor};

use crate::test_utils::assert_tensor;

#[test]
fn test_backprop() -> Result<()> {
    let device = Device::get_cpu_device();
    let x = tensor::Tensor::from_scalar(2.0, device).requires_grad();
    let a = x.pow_scalar(2.0)?;
    let y = (a.pow_scalar(2.0)? + a.pow_scalar(2.0)?)?;
    assert_tensor(&y, &ten![32.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![64.0]);
    Ok(())
}
