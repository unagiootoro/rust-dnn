use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::Layer;

use crate::{
    time_embedding::{self, TimeEmbedding},
    unet::UNET,
    unet_output_layer::UNET_OutputLayer,
};

pub struct Diffusion<B: Backend> {
    time_embedding: TimeEmbedding<B>,
    unet: UNET<B>,
    final_layer: UNET_OutputLayer<B>,
}

impl<B: Backend> Diffusion<B> {
    pub fn new(device: Device<B>) -> Self {
        let time_embedding = TimeEmbedding::new(320, device);
        let unet = UNET::new(device);
        let final_layer = UNET_OutputLayer::new(320, 4, device);
        Self {
            time_embedding,
            unet,
            final_layer,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>, context: &Tensor<B, f32>, time: &Tensor<B, f32>) -> Tensor<B, f32> {
        let time = self.time_embedding.forward(&time);
        let output = self.unet.forward(&x, &context, &time);
        let output = self.final_layer.forward(&output);
        output
    }
}

impl<B: Backend> Layer<B, f32> for Diffusion<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}
