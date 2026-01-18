use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend,
    device::{self, Device},
    tensor::Tensor,
};
use rust_dnn_nn::{
    layer::{Conv2D, GroupNorm, Layer},
    layer_list::LayerList,
    sequential::{DynSequential, SequentialItem},
};

use crate::self_attention::SelfAttention;

pub struct VAE_AttentionBlock<B: Backend> {
    groupnorm: GroupNorm<B, f32>,
    attention: SelfAttention<B>,
}

impl<B: Backend> VAE_AttentionBlock<B> {
    pub fn new(channels: usize, device: Device<B>) -> Self {
        let groupnorm = GroupNorm::new(32, channels, 1e-7, true, device);
        let attention = SelfAttention::new(1, channels, true, true, device);
        Self {
            groupnorm,
            attention,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        let residue = x.clone();
        let x = self.groupnorm.forward(x);
        let n = x.size(0);
        let c = x.size(1);
        let h = x.size(2);
        let w = x.size(3);

        let x = x.reshape(vec![n, c, h * w]);
        let x = x.transpose(-1, -2);
        let x = self.attention.forward(&x, false);
        let x = x.transpose(-1, -2);
        let x = x.reshape(vec![n, c, h, w]);

        let x = x + residue;
        x
    }
}

impl<B: Backend> Layer<B, f32> for VAE_AttentionBlock<B> {}

impl<B: Backend> SequentialItem<B, f32> for VAE_AttentionBlock<B> {
    fn forward(&mut self, x: Tensor<B, f32>, _is_train: bool) -> Tensor<B, f32> {
        VAE_AttentionBlock::forward(&self, &x)
    }
}

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

impl<B: Backend> Layer<B, f32> for VAE_ResidualBlock<B> {}

impl<B: Backend> SequentialItem<B, f32> for VAE_ResidualBlock<B> {
    fn forward(&mut self, x: Tensor<B, f32>, is_train: bool) -> Tensor<B, f32> {
        VAE_ResidualBlock::forward(&self, x)
    }
}

pub struct VAE_Decoder<B: Backend + 'static> {
    pub sequential: DynSequential<B, f32>,
}

impl<B: Backend> VAE_Decoder<B> {
    pub fn new(device: Device<B>) -> Self {
        let mut sequential = DynSequential::new();

        //             # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        //             nn.Conv2d(4, 4, kernel_size=1, padding=0),
        sequential.add(Conv2D::new(
            4,
            4,
            1,
            1,
            1,
            1,
            Some((0, 0)),
            false,
            true,
            device,
        ));

        //             # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             nn.Conv2d(4, 512, kernel_size=3, padding=1),

        sequential.add(Conv2D::new(
            4,
            512,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        ));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_AttentionBlock(512),

        sequential.add(VAE_AttentionBlock::new(512, device));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
        //             # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
        //             nn.Upsample(scale_factor=2),

        // TODO: Upsample

        //             # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        //             nn.Conv2d(512, 512, kernel_size=3, padding=1),

        sequential.add(Conv2D::new(
            512,
            512,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        ));

        //             # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
        //             VAE_ResidualBlock(512, 512),

        sequential.add(VAE_ResidualBlock::new(512, 512, device));

        //             # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
        //             nn.Upsample(scale_factor=2),

        // TODO: Upsample

        //             # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
        //             nn.Conv2d(512, 512, kernel_size=3, padding=1),

        sequential.add(Conv2D::new(
            512,
            256,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        ));

        //             # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        //             VAE_ResidualBlock(512, 256),

        sequential.add(VAE_ResidualBlock::new(512, 256, device));

        //             # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        //             VAE_ResidualBlock(256, 256),

        sequential.add(VAE_ResidualBlock::new(256, 256, device));

        //             # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
        //             VAE_ResidualBlock(256, 256),

        sequential.add(VAE_ResidualBlock::new(256, 256, device));

        //             # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
        //             nn.Upsample(scale_factor=2),

        // TODO: Upsample

        //             # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
        //             nn.Conv2d(256, 256, kernel_size=3, padding=1),

        sequential.add(Conv2D::new(
            256,
            256,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        ));

        //             # (Batch_Size, 256, Height, Width) -> (Batch_Size, 128, Height, Width)
        //             VAE_ResidualBlock(256, 128),

        sequential.add(VAE_ResidualBlock::new(256, 128, device));

        //             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        //             VAE_ResidualBlock(128, 128),

        sequential.add(VAE_ResidualBlock::new(128, 128, device));

        //             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        //             VAE_ResidualBlock(128, 128),

        sequential.add(VAE_ResidualBlock::new(128, 128, device));

        //             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        //             nn.GroupNorm(32, 128),

        sequential.add(GroupNorm::new(32, 128, 1e-7, true, device));

        //             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
        //             nn.SiLU(),

        // TODO: SiLU

        //             # (Batch_Size, 128, Height, Width) -> (Batch_Size, 3, Height, Width)
        //             nn.Conv2d(128, 3, kernel_size=3, padding=1),

        sequential.add(Conv2D::new(
            128,
            3,
            3,
            3,
            1,
            1,
            Some((1, 1)),
            false,
            true,
            device,
        ));

        Self { sequential }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        let x = x / 0.18215;
        self.sequential.forward(x, false)
    }
}

impl<B: Backend> Layer<B, f32> for VAE_Decoder<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        self.sequential.layers_map()
    }
}
