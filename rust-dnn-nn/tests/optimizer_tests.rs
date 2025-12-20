mod test_utils;

use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::{
    layer::{Layer, Linear},
    loss::mean_squared_error,
    optimizer::{Adam, AdamW, Optimizer, SGD},
};

use crate::test_utils::assert_tensor;

fn test_sgd_update_parameters<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(0..(2 * 3), device).reshape(vec![2, 3]);
    let w = Tensor::arange(0..(1 * 3), device)
        .reshape(vec![1, 3])
        .requires_grad();
    let b = Tensor::from_scalar(0.5, device).requires_grad();
    let t = Tensor::from_vec(vec![1., 0.], vec![2, 1], device);
    let linear = Linear::from_weights(w.clone(), Some(b.clone()));
    let y = linear.forward(&x);

    let loss = mean_squared_error(&y, &t)?;
    assert_eq!(loss.to_vec(), vec![115.25]);
    let grads = loss.backward();

    let mut optimizer = SGD::new(0.01);
    optimizer.update_parameters(&mut linear.all_parameters_map(), &grads);
    assert_tensor(&w, &ten![[-0.4350, 0.3750, 1.1850]]);
    assert_tensor(&b, &ten![0.3100]);

    let y = linear.forward(&x);
    let loss: Tensor<B, f64> = mean_squared_error(&y, &t)?;
    assert!((loss.to_vec()[0] - 22.783962500000005).abs() < 1e-4);
    let grads = loss.backward();
    optimizer.update_parameters(&mut linear.all_parameters_map(), &grads);
    assert_tensor(&w, &ten![[-0.6279, 0.0973, 0.8224]]);
    assert_tensor(&b, &ten![0.2252]);

    Ok(())
}

define_test!(
    test_sgd_update_parameters,
    test_sgd_update_parameters_cpu,
    test_sgd_update_parameters_cuda
);

fn test_adam_update_parameters<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(0..(2 * 3), device).reshape(vec![2, 3]);
    let w = Tensor::arange(0..(1 * 3), device)
        .reshape(vec![1, 3])
        .requires_grad();
    let b = Tensor::from_scalar(0.5, device).requires_grad();
    let t = Tensor::from_vec(vec![1., 0.], vec![2, 1], device);

    let linear = Linear::from_weights(w.clone(), Some(b.clone()));
    let y = linear.forward(&x);

    let loss = mean_squared_error(&y, &t)?;
    assert_eq!(loss.to_vec(), vec![115.25]);
    let grads = loss.backward();

    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
    {
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), w.clone());
        parameters.insert("b".to_string(), b.clone());
        optimizer.update_parameters(&mut parameters, &grads);
    }
    assert_tensor(&w, &ten![[-0.0100, 0.9900, 1.9900]]);
    assert_tensor(&b, &ten![0.49]);

    let y = linear.forward(&x);
    let loss = mean_squared_error(&y, &t)?;
    let grads = loss.backward();
    {
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), w.clone());
        parameters.insert("b".to_string(), b.clone());
        optimizer.update_parameters(&mut parameters, &grads);
    }
    assert_tensor(&w, &ten![[-0.0200, 0.9800, 1.9800]]);
    assert_tensor(&b, &ten![0.4800]);

    Ok(())
}

define_test!(
    test_adam_update_parameters,
    test_adam_update_parameters_cpu,
    test_adam_update_parameters_cuda
);

fn test_adamw_update_parameters<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::arange(0..(2 * 3), device).reshape(vec![2, 3]);
    let w = Tensor::arange(0..(1 * 3), device)
        .reshape(vec![1, 3])
        .requires_grad();
    let b = Tensor::from_scalar(0.5, device).requires_grad();
    let t = Tensor::from_vec(vec![1., 0.], vec![2, 1], device);

    let linear = Linear::from_weights(w.clone(), Some(b.clone()));
    let y = linear.forward(&x);

    let loss = mean_squared_error(&y, &t)?;
    assert_eq!(loss.to_vec(), vec![115.25]);
    let grads = loss.backward();

    let mut optimizer = AdamW::new(0.01, 0.9, 0.999, 1e-8, 1e-2);
    {
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), w.clone());
        parameters.insert("b".to_string(), b.clone());
        optimizer.update_parameters(&mut parameters, &grads);
    }
    assert_tensor(&w, &ten![[-0.0100, 0.9899, 1.9898]]);
    assert_tensor(&b, &ten![0.4900]);

    let y = linear.forward(&x);
    let loss = mean_squared_error(&y, &t)?;
    let grads = loss.backward();
    {
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), w.clone());
        parameters.insert("b".to_string(), b.clone());
        optimizer.update_parameters(&mut parameters, &grads);
    }
    assert_tensor(&w, &ten![[-0.0200, 0.9798, 1.9796]]);
    assert_tensor(&b, &ten![0.4799]);

    Ok(())
}

define_test!(
    test_adamw_update_parameters,
    test_adamw_update_parameters_cpu,
    test_adamw_update_parameters_cuda
);
