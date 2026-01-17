// class SwitchSequential(nn.Sequential):
//     def forward(self, x, context, time):
//         for layer in self:
//             if isinstance(layer, UNET_AttentionBlock):
//                 x = layer(x, context)
//             elif isinstance(layer, UNET_ResidualBlock):
//                 x = layer(x, time)
//             else:
//                 x = layer(x)
//         return x

use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::{layer::Layer, layer_list::DynLayerList};


pub struct SwitchSequential<B: Backend + 'static> {
    pub layer_list: DynLayerList<B, f32>,
}

impl<B: Backend> SwitchSequential<B> {
    pub fn new(device: Device<B>) -> Self {
        Self { layer_list: DynLayerList::new() }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, context: &Tensor<B, f32>, time: &Tensor<B, f32>) -> Tensor<B, f32> {
        todo!()
    }
}

impl<B: Backend> Layer<B, f32> for SwitchSequential<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        self.layer_list.layers_map()
    }
}
