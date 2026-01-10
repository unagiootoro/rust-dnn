use core::f32;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::Linear;

pub struct CrossAttention<B: Backend> {
    q_proj: Linear<B, f32>,
    k_proj: Linear<B, f32>,
    v_proj: Linear<B, f32>,
    out_proj: Linear<B, f32>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(
        n_heads: usize,
        d_embed: usize,
        d_cross: usize,
        in_proj_bias: bool,
        out_proj_bias: bool,
        device: Device<B>,
    ) -> Self {
        let q_proj = Linear::new(d_embed, d_embed, in_proj_bias, device);
        let k_proj = Linear::new(d_cross, d_embed, in_proj_bias, device);
        let v_proj = Linear::new(d_cross, d_embed, in_proj_bias, device);
        let out_proj = Linear::new(d_embed, d_embed, out_proj_bias, device);
        let d_head = d_embed / n_heads;
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            d_head,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>, causal_mask: bool) -> Tensor<B, f32> {
        let input_shape = x.shape().to_vec();
        let batch_size = input_shape[0];
        let sequence_length = input_shape[1];

        let interim_shape = vec![batch_size, sequence_length, self.n_heads, self.d_head];

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q = q.reshape(interim_shape.clone()).transpose(1, 2);
        let k = k.reshape(interim_shape.clone()).transpose(1, 2);
        let v = v.reshape(interim_shape.clone()).transpose(1, 2);

        let weight = q.matmul(&k.transpose(-1, -2));
        if causal_mask {
            let mask = Tensor::ones(weight.shape().to_vec(), x.device());
            weight.masked_fill(&mask, -f32::MAX);
        }

        let weight = weight / (self.d_head as f64).sqrt();
        let weight = weight.softmax(-1);

        let output = weight.matmul(&v);
        let output = output.transpose(1, 2);
        let output = output.reshape(input_shape);
        self.out_proj.forward(&output)
    }
}
