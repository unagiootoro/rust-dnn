mod test_utils;

use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::{
    layer::linear,
    loss::{cross_entropy, mean_squared_error, sigmoid_cross_entropy},
    optimizer::{Optimizer, SGD},
};

use crate::test_utils::assert_tensor;

fn test_mean_squared_error<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?;
    let t = ten![3.0, 4.0, 5.0].to_device(device)?;
    let y = mean_squared_error(&x, &t)?;
    assert_tensor(&y, &ten![9.0]);
    Ok(())
}

define_test!(
    test_mean_squared_error,
    test_mean_squared_error_cpu,
    test_mean_squared_error_cuda
);

fn test_mean_squared_error_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.0, 1.0, 2.0].to_device(device)?.requires_grad();
    let t = ten![3.0, 4.0, 5.0].to_device(device)?;
    let y = mean_squared_error(&x, &t)?;
    assert_tensor(&y, &ten![9.0]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![-2.0, -2.0, -2.0]);
    Ok(())
}

define_test!(
    test_mean_squared_error_backward,
    test_mean_squared_error_backward_cpu,
    test_mean_squared_error_backward_cuda
);

fn test_cross_entropy<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?.requires_grad();
    let t = ten![2].to_device(device)?;
    let y = cross_entropy(&x, &t)?;
    assert_tensor(&y, &ten![0.4076058]);
    Ok(())
}

define_test!(
    test_cross_entropy,
    test_cross_entropy_cpu,
    test_cross_entropy_cuda
);

fn test_cross_entropy_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![[1.0, 2.0, 3.0]].to_device(device)?.requires_grad();
    let t = ten![2].to_device(device)?;
    let y = cross_entropy(&x, &t)?;
    assert_tensor(&y, &ten![0.4076058]);
    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(gx, &ten![[0.09003057, 0.24472848, -0.33475906]]);
    Ok(())
}

define_test!(
    test_cross_entropy_backward,
    test_cross_entropy_backward_cpu,
    test_cross_entropy_backward_cuda
);

fn test_sigmoid_cross_entropy<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5, 0.0, -0.5];
    let t = ten![1.0, 0.0, 1.0];
    let y = sigmoid_cross_entropy(&x, &t)?;
    assert_tensor(&y, &ten![0.7137670516967773]);
    Ok(())
}

define_test!(
    test_sigmoid_cross_entropy,
    test_sigmoid_cross_entropy_cpu,
    test_sigmoid_cross_entropy_cuda
);

fn test_sigmoid_cross_entropy_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = ten![0.5, 0.0, -0.5].requires_grad();
    let t = ten![1.0, 0.0, 1.0].requires_grad();
    let y = sigmoid_cross_entropy(&x, &t)?;
    assert_tensor(&y, &ten![0.7137670516967773]);

    let grads = y.backward();
    let gx = grads.get(&x).unwrap();
    assert_tensor(&gx, &ten![-0.1258, 0.1667, -0.2075]);
    Ok(())
}

define_test!(
    test_sigmoid_cross_entropy_backward,
    test_sigmoid_cross_entropy_backward_cpu,
    test_sigmoid_cross_entropy_backward_cuda
);
