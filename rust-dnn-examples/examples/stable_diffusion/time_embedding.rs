use core::f32;
use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::{Layer, Linear};

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
        let x = self.linear_2.forward(&x);
        x
    }
}

impl<B: Backend> Layer<B, f32> for TimeEmbedding<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("linear_1".to_string(), &self.linear_1);
        map.insert("linear_2".to_string(), &self.linear_2);
        map
    }
}
