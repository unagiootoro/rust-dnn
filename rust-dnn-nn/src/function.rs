use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, num::Num, tensor::Tensor,
};

pub fn rms_norm<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    eps: f64,
) -> Tensor<B, T> {
    let xn = x * (x.pow_scalar(2.0).mean_axis(-1, true) + eps).rsqrt();
    xn * gamma
}

pub fn scaled_dot_product_attention<B: Backend, T: Float>(
    q: &Tensor<B, T>,
    k: &Tensor<B, T>,
    v: &Tensor<B, T>,
    attn_mask: Option<&Tensor<B, T>>,
    dropout_ratio: f64,
    is_train: bool,
    seed: Option<u64>,
) -> Tensor<B, T> {
    let rhs = k.permuted_axes(&[0, 1, 3, 2]) / (q.shape()[3] as f64).sqrt();
    let scores = q.matmul(&rhs);

    let scores = if let Some(attn_mask) = attn_mask {
        scores.masked_fill(&attn_mask.eq_scalar(0.0), -T::max_value())
    } else {
        scores
    };

    let attn = scores.softmax(3);

    let attn = if dropout_ratio > 0.0 {
        attn.dropout(dropout_ratio, is_train, seed)
    } else {
        attn
    };

    attn.matmul(&v)
}

pub fn generate_causal_attention_mask<B: Backend, T: Num>(
    seq_len: usize,
    device: Device<B>,
) -> Result<Tensor<B, T>> {
    let ones = Tensor::ones(vec![seq_len, seq_len], device);
    Ok(ones.tril())
}
