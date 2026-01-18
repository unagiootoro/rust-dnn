use core::f32;
use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::*;

pub struct UNET_ResidualBlock<B: Backend> {
    groupnorm_feature: GroupNorm<B, f32>,
    conv_feature: Conv2D<B, f32>,
    linear_time: Linear<B, f32>,
    groupnorm_merged: GroupNorm<B, f32>,
    conv_merged: Conv2D<B, f32>,
    residual_layer: Option<Conv2D<B, f32>>,
}

impl<B: Backend> UNET_ResidualBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, device: Device<B>) -> Self {
        Self::new2(in_channels, out_channels, 1280, device)
    }

    pub fn new2(in_channels: usize, out_channels: usize, n_time: usize, device: Device<B>) -> Self {
        let groupnorm_feature = GroupNorm::new(32, in_channels, 1e-5, true, device);
        let conv_feature = Conv2D::new(in_channels, out_channels, 3, 3, 1, 1, Some((1, 1)), false, true, device);
        let linear_time = Linear::new(n_time, out_channels, true, device);
        let groupnorm_merged = GroupNorm::new(32, out_channels, 1e-5, true, device);
        let conv_merged = Conv2D::new(out_channels, out_channels, 3, 3, 1, 1, Some((1, 1)), false, true, device);
        let residual_layer = if in_channels == out_channels {
            None
        } else {
            Some(Conv2D::new(in_channels, out_channels, 1, 1, 1, 1, None, false, true, device))
        };
        Self {
            groupnorm_feature,
            conv_feature,
            linear_time,
            groupnorm_merged,
            conv_merged,
            residual_layer,
        }
    }

    pub fn forward(&self, feature: &Tensor<B, f32>, time: &Tensor<B, f32>) -> Tensor<B, f32> {
        //         # feature: (Batch_Size, In_Channels, Height, Width)
        //         # time: (1, 1280)

        //         residue = feature
        let residue = feature.clone();

        //         # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        //         feature = self.groupnorm_feature(feature)
        let feature = self.groupnorm_feature.forward(&feature);

        //         # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        //         feature = F.silu(feature)
        let feature = feature.silu();

        //         # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        //         feature = self.conv_feature(feature)
        let feature = self.conv_feature.forward(&feature);

        //         # (1, 1280) -> (1, 1280)
        //         time = F.silu(time)
        let time = time.silu();

        //         # (1, 1280) -> (1, Out_Channels)
        //         time = self.linear_time(time)
        let time = self.linear_time.forward(&time);

        //         # Add width and height dimension to time.
        //         # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        //         merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        let merged = feature + time.unsqueeze(-1).unsqueeze(-1);

        //         # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        //         merged = self.groupnorm_merged(merged)
        let merged = self.groupnorm_merged.forward(&merged);

        //         # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        //         merged = F.silu(merged)
        let merged = merged.silu();

        //         # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        //         merged = self.conv_merged(merged)
        let merged = self.conv_merged.forward(&merged);

        //         # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        //         return merged + self.residual_layer(residue)
        if let Some(ref residual_layer) = self.residual_layer {
            merged + residual_layer.forward(&residue)
        } else {
            merged
        }
    }
}

impl<B: Backend> Layer<B, f32> for UNET_ResidualBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}
