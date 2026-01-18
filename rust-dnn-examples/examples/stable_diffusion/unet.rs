use core::f32;
use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::{
    layer::{Conv2D, Layer},
    layer_list::{DynLayerList, LayerList},
};

use crate::{
    switch_sequential::{SwitchSequential, SwitchSequentialItem}, unet_attention_block::UNET_AttentionBlock,
    unet_residual_block::UNET_ResidualBlock, upsampling::Upsample,
};

pub struct UNET<B: Backend> {
    encoders: LayerList<SwitchSequential<B, f32>, B, f32>,
    bottleneck: SwitchSequential<B, f32>,
    decoders: LayerList<SwitchSequential<B, f32>, B, f32>,
}

impl<B: Backend> UNET<B> {
    pub fn new(device: Device<B>) -> Self {
        //         self.encoders = nn.ModuleList([
        //             # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

        let mut encoders = LayerList::new();

        {
            let mut s = SwitchSequential::new();
            s.add(Conv2D::new(
                4,
                320,
                3,
                3,
                1,
                1,
                Some((1, 1)),
                false,
                true,
                device,
            ));
            encoders.add(s);
        }

        //             # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(320, 320, device));
            s.add(UNET_AttentionBlock::new(8, 40, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(320, 320, device));
            s.add(UNET_AttentionBlock::new(8, 40, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
        //             SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
        {
            let mut s = SwitchSequential::new();
            s.add(Conv2D::new(
                320,
                320,
                3,
                3,
                2,
                2,
                Some((1, 1)),
                false,
                true,
                device,
            ));
            encoders.add(s);
        }

        //             # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
        //             SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(320, 640, device));
            s.add(UNET_AttentionBlock::new(8, 80, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
        //             SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(640, 640, device));
            s.add(UNET_AttentionBlock::new(8, 80, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
        //             SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
        {
            let mut s = SwitchSequential::new();
            s.add(Conv2D::new(
                640,
                640,
                3,
                3,
                2,
                2,
                Some((1, 1)),
                false,
                true,
                device,
            ));
            encoders.add(s);
        }

        //             # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
        //             SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(640, 1280, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
        //             SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1280, 1280, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
        {
            let mut s = SwitchSequential::new();
            s.add(Conv2D::new(
                1280,
                1280,
                3,
                3,
                2,
                2,
                Some((1, 1)),
                false,
                true,
                device,
            ));
            encoders.add(s);
        }

        //             # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1280, 1280, device));
            encoders.add(s);
        }

        //             # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        //         ])
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1280, 1280, device));
            encoders.add(s);
        }

        //         self.bottleneck = SwitchSequential(
        //             # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             UNET_ResidualBlock(1280, 1280),

        //             # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             UNET_AttentionBlock(8, 160),

        //             # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             UNET_ResidualBlock(1280, 1280),
        //         )

        let mut bottleneck = SwitchSequential::new();
        bottleneck.add(UNET_ResidualBlock::new(1280, 1280, device));
        bottleneck.add(UNET_AttentionBlock::new(8, 160, device));
        bottleneck.add(UNET_ResidualBlock::new(1280, 1280, device));

        let mut decoders = LayerList::new();
        //         self.decoders = nn.ModuleList([
        //             # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             SwitchSequential(UNET_ResidualBlock(2560, 1280)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(2560, 1280, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
        //             SwitchSequential(UNET_ResidualBlock(2560, 1280)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(2560, 1280, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
        //             SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(2560, 1280, device));
            s.add(Upsample::new(1280, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
        //             SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(2560, 1280, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
        //             SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(2560, 1280, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
        //             SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1920, 1280, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            s.add(Upsample::new(1280, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
        //             SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1920, 640, device));
            s.add(UNET_AttentionBlock::new(8, 160, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
        //             SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(1280, 640, device));
            s.add(UNET_AttentionBlock::new(8, 80, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(960, 640, device));
            s.add(UNET_AttentionBlock::new(8, 80, device));
            s.add(Upsample::new(640, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(960, 640, device));
            s.add(UNET_AttentionBlock::new(8, 40, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(640, 320, device));
            s.add(UNET_AttentionBlock::new(8, 40, device));
            decoders.add(s);
        }

        //             # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        //             SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        //         ])
        {
            let mut s = SwitchSequential::new();
            s.add(UNET_ResidualBlock::new(640, 320, device));
            s.add(UNET_AttentionBlock::new(8, 40, device));
            decoders.add(s);
        }

        Self {
            encoders,
            bottleneck,
            decoders,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>, context: &Tensor<B, f32>, time: &Tensor<B, f32>) -> Tensor<B, f32> {
        let mut x = x.clone();
        let mut skip_connections = Vec::new();
        for layers in self.encoders.layers() {
            x = layers.forward(&x, &context, &time);
            skip_connections.push(x.clone());
        }

        x = self.bottleneck.forward(&x, context, time);

        for layers in self.decoders.layers() {
            x = Tensor::cat(&[x, skip_connections.pop().unwrap()], 1);
            x = layers.forward(&x, &context, &time);
        }

        x
    }
}

impl<B: Backend> Layer<B, f32> for UNET<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("encoders".to_string(), &self.encoders);
        map.insert("bottleneck".to_string(), &self.bottleneck);
        map.insert("decoders".to_string(), &self.decoders);
        map
    }
}
