use core::f32;
use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::{Layer, Linear};

pub struct SelfAttention<B: Backend> {
    in_proj: Linear<B, f32>,
    out_proj: Linear<B, f32>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> SelfAttention<B> {
    pub fn new(
        n_heads: usize,
        d_embed: usize,
        in_proj_bias: bool,
        out_proj_bias: bool,
        device: Device<B>,
    ) -> Self {
        let in_proj = Linear::new(d_embed, 3 * d_embed, in_proj_bias, device);
        let out_proj = Linear::new(d_embed, d_embed, out_proj_bias, device);
        let d_head = d_embed / n_heads;
        Self {
            in_proj,
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

        let h = self.in_proj.forward(x);
        let chunks = h.chunk(-1, 3);
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        let q = q.reshape(interim_shape.clone()).transpose(1, 2);
        let k = k.reshape(interim_shape.clone()).transpose(1, 2);
        let v = v.reshape(interim_shape.clone()).transpose(1, 2);

        let mut weight = q.matmul(&k.transpose(-1, -2));
        if causal_mask {
            let mask = Tensor::ones(weight.shape().to_vec(), x.device()).triu3(1);
            weight = weight.masked_fill(&mask, -f32::MAX);
        }

        let weight = weight / (self.d_head as f64).sqrt();
        let weight = weight.softmax(-1);

        let output = weight.matmul(&v);
        let output = output.transpose(1, 2);
        let output = output.reshape(input_shape);
        let output = self.out_proj.forward(&output);
        output
    }
}

impl<B: Backend> Layer<B, f32> for SelfAttention<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("in_proj".to_string(), &self.in_proj);
        map.insert("out_proj".to_string(), &self.out_proj);
        map
    }
}
