use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::{Conv2D, GroupNorm, Layer};

pub struct UNET_OutputLayer<B: Backend> {
    groupnorm: GroupNorm<B, f32>,
    conv: Conv2D<B, f32>,
}

impl<B: Backend> UNET_OutputLayer<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: Device<B>) -> Self {
        let groupnorm = GroupNorm::new(32, in_channels, 1e-5, true, device);
        let conv = Conv2D::new(
            in_channels,
            out_channels,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        );
        Self { groupnorm, conv }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        //         # x: (Batch_Size, 320, Height / 8, Width / 8)

        //         # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //         x = self.groupnorm(x)
        let x = self.groupnorm.forward(&x);

        //         # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //         x = F.silu(x)
        let x = x.silu();

        //         # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        //         x = self.conv(x)
        let x = self.conv.forward(&x);

        //         # (Batch_Size, 4, Height / 8, Width / 8)
        //         return x
        x
    }
}

impl<B: Backend> Layer<B, f32> for UNET_OutputLayer<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("groupnorm".to_string(), &self.groupnorm);
        map.insert("conv".to_string(), &self.conv);
        map
    }
}
