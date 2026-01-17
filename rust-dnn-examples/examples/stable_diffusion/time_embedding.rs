use core::f32;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::Linear;

pub struct TimeEmbedding<B: Backend> {
    linear_1: Linear<B, f32>,
    linear_2: Linear<B, f32>,
}

impl<B: Backend> TimeEmbedding<B> {
    pub fn new(n_embd: usize, device: Device<B>) -> Self {
        let linear_1 = Linear::new(n_embd * 4, n_embd, true, device);
        let linear_2 = Linear::new(4 * n_embd, 4 * n_embd, true, device);
        Self { linear_1, linear_2 }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        let x = self.linear_1.forward(x);
        let x = x.silu();
        let x = self.linear_2.forward(x);
        x
    }
}
