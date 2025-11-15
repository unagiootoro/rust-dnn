mod test_utils;

use crate::test_utils::assert_tensor;
use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::layer::{BatchNorm1d, Linear};

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

define_test!(
    test_linear_forward,
    test_linear_forward_cpu,
    test_linear_forward_cuda
);

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

define_test!(
    test_linear_no_bias_forward,
    test_linear_no_bias_forward_cpu,
    test_linear_no_bias_forward_cuda
);

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

define_test!(
    test_linear_backward,
    test_linear_backward_cpu,
    test_linear_backward_cuda
);

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
    test_linear_no_bias_backward_cpu,
    test_linear_no_bias_backward_cuda
);

fn test_batch_norm1d<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 1.0, 3.0, 7.0, 8.0],
        vec![4, 3],
        device,
    )?
    .requires_grad();
    let gamma = Tensor::from_vec(vec![1.0, 1.5, 2.0], vec![3], device)?.requires_grad();
    let beta = Tensor::from_vec(vec![0.0, 0.5, -1.0], vec![3], device)?.requires_grad();
    let mut batch_norm1d = BatchNorm1d::new(3, 0.9, 1e-7, device);
    batch_norm1d.gamma().copy(&gamma)?;
    batch_norm1d.beta().copy(&beta)?;
    let y = batch_norm1d.forward(&x, true)?;
    assert_tensor(
        &y,
        &ten![
            [-1.3416, 1.2862, -2.1142],
            [-0.4472, -1.2297, 0.1142],
            [1.3416, -0.6007, -3.5997],
            [0.4472, 2.5442, 1.5997,]
        ],
    );
    Ok(())
}

define_test!(
    test_batch_norm1d,
    test_batch_norm1d_cpu,
    test_batch_norm1d_cuda
);

fn test_batch_norm1d_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(
        vec![1.0, 5.0, 3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 1.0, 3.0, 7.0, 8.0],
        vec![4, 3],
        device,
    )?
    .requires_grad();
    let gamma = Tensor::from_vec(vec![1.0, 1.5, 2.0], vec![3], device)?;
    let beta = Tensor::from_vec(vec![0.0, 0.5, -1.0], vec![3], device)?;
    let mut batch_norm1d = BatchNorm1d::new(3, 0.9, 1e-7, device);
    batch_norm1d.gamma().copy(&gamma)?;
    batch_norm1d.beta().copy(&beta)?;
    let y = batch_norm1d.forward(&x, true)?;
    assert_tensor(
        &y,
        &ten![
            [-1.3416, 1.2862, -2.1142],
            [-0.4472, -1.2297, 0.1142],
            [1.3416, -0.6007, -3.5997],
            [0.4472, 2.5442, 1.5997,]
        ],
    );

    let x2 = Tensor::from_vec(
        vec![
            1.0, 0.0, -1.0, -1.0, 2.0, 0.0, 0.5, -0.5, 1.0, -1.5, 1.0, -0.5,
        ],
        vec![4, 3],
        device,
    )?;
    let y = (&y * x2)?;

    let grads = y.backward()?;
    let gx = grads.get(&x).unwrap();
    let ggamma = grads.get(&batch_norm1d.gamma()).unwrap();
    let gbeta = grads.get(&batch_norm1d.beta()).unwrap();
    assert_tensor(
        &gx,
        &ten![
            [0.8497, -0.3456, -0.7940],
            [-0.7603, 0.7603, 0.2369],
            [0.9391, -0.7741, 0.4995],
            [-1.0286, 0.3594, 0.0576],
        ],
    );
    assert_tensor(&ggamma, &ten![-0.8944, -0.5766, -1.3927]);
    assert_tensor(&gbeta, &ten![-1.0000, 2.5000, -0.5000]);
    Ok(())
}

define_test!(
    test_batch_norm1d_backward,
    test_batch_norm1d_backward_cpu,
    test_batch_norm1d_backward_cuda
);
