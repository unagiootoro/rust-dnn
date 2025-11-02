mod test_utils;

use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::{
    layer::{Layer, Linear},
    loss::mean_squared_error,
    optimizer::{Optimizer, SGD},
};

use crate::test_utils::assert_tensor;

fn test_sgd_update_parameters<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(0..(2 * 3), device).reshape(vec![2, 3])?;
    let w = Tensor::arange(0..(1 * 3), device)
        .reshape(vec![1, 3])?
        .requires_grad();
    let b = Tensor::from_scalar(0.5, device).requires_grad();
    let t = Tensor::from_vec(vec![1., 0.], vec![2, 1], device)?;
    let linear = Linear::from_weights(w.clone(), Some(b.clone()))?;
    let y = linear.forward(&x)?;

    let loss = mean_squared_error(&y, &t)?;
    assert_eq!(loss.to_vec()?, vec![115.25]);
    let grads = loss.backward()?;

    let mut optimizer = SGD::new(0.01);
    optimizer.update_parameters(&mut linear.all_parameters_map(), &grads)?;
    assert_tensor(&w, &ten![[-0.4350, 0.3750, 1.1850]]);
    assert_tensor(&b, &ten![0.3100]);

    let y = linear.forward(&x)?;
    let loss: Tensor<B, f64> = mean_squared_error(&y, &t)?;
    assert_eq!(loss.to_vec()?, vec![22.783962500000005]);
    let grads = loss.backward()?;
    optimizer.update_parameters(&mut linear.all_parameters_map(), &grads)?;
    assert_tensor(&w, &ten![[-0.6279, 0.0973, 0.8224]]);
    assert_tensor(&b, &ten![0.2252]);

    Ok(())
}

define_test!(test_sgd_update_parameters, test_sgd_update_parameters_cpu);
