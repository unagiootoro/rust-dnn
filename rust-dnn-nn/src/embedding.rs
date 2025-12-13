use std::collections::HashMap;

use rust_dnn_core::backend::Backend;
use rust_dnn_core::device::Device;
use rust_dnn_core::error::Result;
use rust_dnn_core::float::Float;
use rust_dnn_core::tensor::Tensor;

use crate::layer::Layer;

pub struct Embedding<B: Backend, T: Float> {
    weight: Tensor<B, T>,
}

impl<B: Backend, T: Float> Layer<B, T> for Embedding<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::<String, Tensor<B, T>>::new();
        map.insert("weight".to_string(), self.weight.clone());
        map
    }
}

impl<B: Backend, T: Float> Embedding<B, T> {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: Device<B>) -> Result<Self> {
        let weight =
            Tensor::rand_norm(&[num_embeddings, embedding_dim], None, device).requires_grad();
        Ok(Self { weight })
    }

    pub fn from_weights(weight: Tensor<B, T>) -> Result<Self> {
        Ok(Self { weight })
    }

    pub fn weight(&self) -> &Tensor<B, T> {
        &self.weight
    }

    pub fn forward(&self, x: &Tensor<B, u32>) -> Result<Tensor<B, T>> {
        embedding::<B, T>(x, &self.weight)
    }
}

pub fn embedding<B: Backend, T: Float>(
    x: &Tensor<B, u32>,
    weight: &Tensor<B, T>,
) -> Result<Tensor<B, T>> {
    let mut output_shape = x.shape().to_vec();
    for i in 1..weight.ndim() {
        output_shape.push(weight.shape()[i]);
    }
    weight.index_select(0, x)?.reshape(output_shape)
}
