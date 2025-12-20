use rust_dnn_core::{backend::Backend, error::Result, float::Float, tensor::Tensor};

pub fn mean_squared_error<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    y: &Tensor<B, T>,
) -> Result<Tensor<B, T>> {
    let diff = x - y;
    let output = diff.pow_scalar(2.0).sum() / diff.len() as f64;
    Ok(output)
}

pub fn cross_entropy<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    y: &Tensor<B, u32>,
) -> Result<Tensor<B, T>> {
    let x = x.log_softmax(1);
    let index = y.unsqueeze(1);
    let output = -x.gather(&index, 1).mean_axis(0, false).sum();
    Ok(output)
}

pub fn sigmoid_cross_entropy<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    y: &Tensor<B, T>,
) -> Result<Tensor<B, T>> {
    let x = x.sigmoid();
    let eps = 1e-7;
    let output = -(y * ((&x + eps).ln()) + ((1.0 - y) * (1.0 - &x + eps).ln()))
        .mean_axis(0, false)
        .sum();
    Ok(output)
}
