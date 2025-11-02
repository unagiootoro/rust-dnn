mod test_utils;

use crate::test_utils::assert_tensor;
use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten};
use rust_dnn_nn::layer::Linear;

fn test_linear_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let bias = ten![1.0, 2.0].to_device(device)?.requires_grad();
    let linear = Linear::from_weights(weight, Some(bias))?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-15.0, -32.0], [15.0, 34.0]]);
    Ok(())
}

define_test!(test_linear_forward, test_linear_forward_cpu);

fn test_linear_no_bias_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let x: rust_dnn_core::tensor::Tensor<B, f64> = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let linear = Linear::from_weights(weight, None)?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-16.0, -34.0], [14.0, 32.0]]);
    Ok(())
}

define_test!(test_linear_no_bias_forward, test_linear_no_bias_forward_cpu);

fn test_linear_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let bias = ten![1.0, 2.0].to_device(device)?.requires_grad();
    let linear = Linear::from_weights(weight.clone(), Some(bias.clone()))?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-15.0, -32.0], [15.0, 34.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let gw = grads.get(&weight).unwrap();
    let gb = grads.get(&bias).unwrap();
    assert_tensor(&gx, &ten![[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
    assert_tensor(&gw, &ten![[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]);
    assert_tensor(&gb, &ten![2.0, 2.0]);
    Ok(())
}

define_test!(test_linear_backward, test_linear_backward_cpu);

fn test_linear_no_bias_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[0.0, -2.0, -4.0], [1.0, 2.0, 3.0]]
        .to_device(device)?
        .requires_grad();
    let weight = ten![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        .to_device(device)?
        .requires_grad();
    let linear = Linear::from_weights(weight.clone(), None)?;
    let y = linear.forward(&x)?;
    assert_tensor(&y, &ten![[-16.0, -34.0], [14.0, 32.0]]);
    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let gw = grads.get(&weight).unwrap();
    assert_tensor(&gx, &ten![[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]);
    assert_tensor(&gw, &ten![[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]);
    Ok(())
}

define_test!(
    test_linear_no_bias_backward,
    test_linear_no_bias_backward_cpu
);
