mod test_utils;
use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::{self, Tensor}};

use crate::test_utils::assert_tensor;

fn test_backprop<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_scalar(2.0, device).requires_grad();
    let a = x.pow_scalar(2.0)?;
    let y = (a.pow_scalar(2.0)? + a.pow_scalar(2.0)?)?;
    assert_tensor(&y, &ten![32.0]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![64.0]);
    Ok(())
}

define_test!(test_backprop, test_backprop_cpu, test_backprop_cuda);
