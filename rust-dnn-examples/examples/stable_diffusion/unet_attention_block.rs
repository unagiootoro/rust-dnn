use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::{Conv2D, GroupNorm, Layer, LayerNorm, Linear};

use crate::{cross_attention::CrossAttention, self_attention::SelfAttention};

pub struct UNET_AttentionBlock<B: Backend> {
    groupnorm: GroupNorm<B, f32>,
    conv_input: Conv2D<B, f32>,
    layernorm_1: LayerNorm<B, f32>,
    attention_1: SelfAttention<B>,
    layernorm_2: LayerNorm<B, f32>,
    attention_2: CrossAttention<B>,
    layernorm_3: LayerNorm<B, f32>,
    linear_geglu_1: Linear<B, f32>,
    linear_geglu_2: Linear<B, f32>,
    conv_output: Conv2D<B, f32>,
}

impl<B: Backend> UNET_AttentionBlock<B> {
    pub fn new(n_head: usize, n_embd: usize, device: Device<B>) -> Self {
        Self::new2(n_head, n_embd, 768, device)
    }

    pub fn new2(n_head: usize, n_embd: usize, d_context: usize, device: Device<B>) -> Self {
        let channels = n_head * n_embd;
        let groupnorm = GroupNorm::new(32, channels, 1e-6, true, device);
        let conv_input = Conv2D::new(channels, channels, 1, 1, 1, 1, None, false, true, device);
        let layernorm_1 = LayerNorm::new(vec![channels], 1e-5, true, device);
        let attention_1 = SelfAttention::new(n_head, channels, false, true, device);
        let layernorm_2 = LayerNorm::new(vec![channels], 1e-5, true, device);
        let attention_2 = CrossAttention::new(n_head, channels, d_context, false, true, device);
        let layernorm_3 = LayerNorm::new(vec![channels], 1e-5, true, device);
        let linear_geglu_1 = Linear::new(channels, 4 * channels * 2, true, device);
        let linear_geglu_2 = Linear::new(4 * channels, channels, true, device);
        let conv_output = Conv2D::new(channels, channels, 1, 1, 1, 1, None, false, true, device);
        Self {
            groupnorm,
            conv_input,
            layernorm_1,
            attention_1,
            layernorm_2,
            attention_2,
            layernorm_3,
            linear_geglu_1,
            linear_geglu_2,
            conv_output,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>, context: &Tensor<B, f32>) -> Tensor<B, f32> {
        //         # x: (Batch_Size, Features, Height, Width)
        //         # context: (Batch_Size, Seq_Len, Dim)

        //         residue_long = x
        let residue_long = x.clone();

        //         # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        //         x = self.groupnorm(x)
        let x = self.groupnorm.forward(x);

        //         # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        //         x = self.conv_input(x)
        let x = self.conv_input.forward(&x);

        //         n, c, h, w = x.shape
        let n = x.size(0);
        let c = x.size(1);
        let h = x.size(2);
        let w = x.size(3);

        //         # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        //         x = x.view((n, c, h * w))
        let x = x.reshape(vec![n, c, h * w]);

        //         # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        //         x = x.transpose(-1, -2)
        let x = x.transpose(-1, -2);

        //         # Normalization + Self-Attention with skip connection

        //         # (Batch_Size, Height * Width, Features)
        //         residue_short = x
        let residue_short = x.clone();

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x = self.layernorm_1(x)
        let x = self.layernorm_1.forward(&x);

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x = self.attention_1(x)
        let x = self.attention_1.forward(&x, false);

        //         # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x += residue_short
        let x = x + residue_short;

        //         # (Batch_Size, Height * Width, Features)
        //         residue_short = x
        let residue_short = x.clone();

        //         # Normalization + Cross-Attention with skip connection

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x = self.layernorm_2(x)
        let x = self.layernorm_2.forward(&x);

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x = self.attention_2(x, context)
        // TODO:
        // let x = self.attention_2.forward(&x, context);

        //         # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x += residue_short
        let x = x + residue_short;

        //         # (Batch_Size, Height * Width, Features)
        //         residue_short = x
        let residue_short = x.clone();

        //         # Normalization + FFN with GeGLU and skip connection

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x = self.layernorm_3(x)
        let x = self.layernorm_3.forward(&x);

        //         # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        //         # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        //         x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        // TODO:

        //         # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        //         x = x * F.gelu(gate)
        // TODO:

        //         # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        //         x = self.linear_geglu_2(x)
        let x = self.linear_geglu_1.forward(&x);

        //         # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        //         x += residue_short
        let x = x + residue_short;

        //         # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        //         x = x.transpose(-1, -2)
        let x = x.transpose(-1, -2);

        //         # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        //         x = x.view((n, c, h, w))
        let x = x.reshape(vec![n, c, h, w]);

        //         # Final skip connection between initial input and output of the block
        //         # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        //         return self.conv_output(x) + residue_long
        self.conv_output.forward(&x) + residue_long
    }
}

impl<B: Backend> Layer<B, f32> for UNET_AttentionBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}
