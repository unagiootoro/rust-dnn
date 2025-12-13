mod test_utils;

use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten, tensor::Tensor};
use rust_dnn_nn::{
    embedding::Embedding,
    function::generate_causal_attention_mask,
    layer::{Layer, Linear},
    multi_head_attention::MultiHeadAttention,
    optimizer::{Optimizer, SGD},
};

use crate::test_utils::{arange_with_shape, assert_tensor};

fn test_embedding_forward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1, 2, 4, 5, 4, 3, 2, 9], vec![2, 4], device)?;
    let w = arange_with_shape::<_, f64>(&[10, 3], device);
    let embedding = Embedding::new(10, 3, device)?;
    embedding.weight().copy(&w)?;
    let y = embedding.forward(&x)?;
    let expected_y = Tensor::from_vec(
        vec![
            3., 4., 5., 6., 7., 8., 12., 13., 14., 15., 16., 17., 12., 13., 14., 9., 10., 11., 6.,
            7., 8., 27., 28., 29.,
        ],
        vec![2, 4, 3],
        device,
    )?;
    assert_tensor(&y, &expected_y);
    Ok(())
}

define_test!(
    test_embedding_forward,
    test_embedding_forward_cpu,
    test_embedding_forward_cuda
);

fn test_embedding_forward2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1, 2, 4, 5, 4, 3, 2, 9, 0, 8, 6, 7], vec![12], device)?;
    let w = arange_with_shape::<_, f64>(&[10, 3], device);
    let embedding = Embedding::new(10, 3, device)?;
    embedding.weight().copy(&w)?;
    let y = embedding.forward(&x)?;
    let expected_y = Tensor::from_vec(
        vec![
            3., 4., 5., 6., 7., 8., 12., 13., 14., 15., 16., 17., 12., 13., 14., 9., 10., 11., 6.,
            7., 8., 27., 28., 29., 0., 1., 2., 24., 25., 26., 18., 19., 20., 21., 22., 23.,
        ],
        vec![12, 3],
        device,
    )?;
    assert_tensor(&y, &expected_y);
    Ok(())
}

define_test!(
    test_embedding_forward2,
    test_embedding_forward2_cpu,
    test_embedding_forward2_cuda
);

fn test_embedding_backward<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1, 2, 4, 5, 4, 3, 2, 9], vec![2, 4], device)?;
    let w = arange_with_shape::<_, f64>(&[10, 3], device);
    let embedding = Embedding::new(10, 3, device)?;
    embedding.weight().copy(&w)?;
    let y = embedding.forward(&x)?;
    let expected_y = Tensor::from_vec(
        vec![
            3., 4., 5., 6., 7., 8., 12., 13., 14., 15., 16., 17., 12., 13., 14., 9., 10., 11., 6.,
            7., 8., 27., 28., 29.,
        ],
        vec![2, 4, 3],
        device,
    )?;
    assert_tensor(&y, &expected_y);

    let sum = y.sum()?;
    let grads = sum.backward()?;
    let gw = grads.get(&embedding.weight()).unwrap();
    let expected_gw = Tensor::from_vec(
        vec![
            0., 0., 0., 1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 1., 1.,
        ],
        vec![10, 3],
        device,
    )?;
    assert_tensor(&gw, &expected_gw);
    Ok(())
}

define_test!(
    test_embedding_backward,
    test_embedding_backward_cpu,
    test_embedding_backward_cuda
);

fn test_embedding_backward2<B: Backend>(device: Device<B>) -> Result<()> {
    let x = Tensor::from_vec(vec![1, 2, 4, 5, 4, 3, 2, 9, 0, 8, 6, 7], vec![12], device)?;
    let w = arange_with_shape::<_, f64>(&[10, 3], device);
    let embedding = Embedding::new(10, 3, device)?;
    embedding.weight().copy(&w)?;
    let y = embedding.forward(&x)?;
    let expected_y = Tensor::from_vec(
        vec![
            3., 4., 5., 6., 7., 8., 12., 13., 14., 15., 16., 17., 12., 13., 14., 9., 10., 11., 6.,
            7., 8., 27., 28., 29., 0., 1., 2., 24., 25., 26., 18., 19., 20., 21., 22., 23.,
        ],
        vec![12, 3],
        device,
    )?;
    assert_tensor(&y, &expected_y);

    let sum = y.sum()?;
    let grads = sum.backward()?;
    let gw = grads.get(&embedding.weight()).unwrap();
    let expected_gw = Tensor::from_vec(
        vec![
            1., 1., 1., 1., 1., 1., 2., 2., 2., 1., 1., 1., 2., 2., 2., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1.,
        ],
        vec![10, 3],
        device,
    )?;
    assert_tensor(&gw, &expected_gw);
    Ok(())
}

define_test!(
    test_embedding_backward2,
    test_embedding_backward2_cpu,
    test_embedding_backward2_cuda
);
