use core::f32;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::Conv2D;

pub struct Upsample<B: Backend> {
    conv: Conv2D<B, f32>,
}

impl<B: Backend> Upsample<B> {
    pub fn new(channels: usize, device: Device<B>) -> Self {
        let conv = Conv2D::new(
            channels,
            channels,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        );
        Self { conv }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        //         # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        //         x = F.interpolate(x, scale_factor=2, mode='nearest')
        // TODO:

        self.conv.forward(&x)
    }
}
