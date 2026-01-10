use rust_dnn_core::{
    backend::Backend,
    device::{self, Device},
    tensor::Tensor,
};
use rust_dnn_nn::layer::{Conv2D, GroupNorm};

pub struct VAE_ResidualBlock<B: Backend> {
    groupnorm_1: GroupNorm<B, f32>,
    conv_1: Conv2D<B, f32>,
    groupnorm_2: GroupNorm<B, f32>,
    conv_2: Conv2D<B, f32>,
    residual_layer: Option<Conv2D<B, f32>>,
}

impl<B: Backend> VAE_ResidualBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: Device<B>) -> Self {
        let groupnorm_1 = GroupNorm::new(32, in_channels, 1e-7, true, device);
        let conv_1 = Conv2D::new(
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
        let groupnorm_2 = GroupNorm::new(32, out_channels, 1e-7, true, device);
        let conv_2 = Conv2D::new(
            out_channels,
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
        let residual_layer = if in_channels == out_channels {
            None
        } else {
            Some(Conv2D::new(
                in_channels,
                out_channels,
                1,
                1,
                1,
                1,
                None,
                false,
                true,
                device,
            ))
        };
        Self {
            groupnorm_1,
            conv_1,
            groupnorm_2,
            conv_2,
            residual_layer,
        }
    }

    pub fn forward(&self, x: Tensor<B, f32>) -> Tensor<B, f32> {
        let residue = x.clone();
        let x = self.groupnorm_1.forward(&x);
        let x = x.silu();
        let x = self.conv_1.forward(&x);
        let x = self.groupnorm_2.forward(&x);
        let x = x.silu();
        let x = self.conv_2.forward(&x);
        if let Some(ref residual_layer) = self.residual_layer {
            x + residual_layer.forward(&residue)
        } else {
            x
        }
    }
}
