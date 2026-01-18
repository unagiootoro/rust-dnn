use core::f32;
use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};
use rust_dnn_nn::layer::{Layer, Linear};

pub struct CrossAttention<B: Backend> {
    q_proj: Linear<B, f32>,
    k_proj: Linear<B, f32>,
    v_proj: Linear<B, f32>,
    out_proj: Linear<B, f32>,
    n_heads: usize,
    d_head: usize,
}

impl<B: Backend> CrossAttention<B> {
    pub fn new(
        n_heads: usize,
        d_embed: usize,
        d_cross: usize,
        in_proj_bias: bool,
        out_proj_bias: bool,
        device: Device<B>,
    ) -> Self {
        let q_proj = Linear::new(d_embed, d_embed, in_proj_bias, device);
        let k_proj = Linear::new(d_cross, d_embed, in_proj_bias, device);
        let v_proj = Linear::new(d_cross, d_embed, in_proj_bias, device);
        let out_proj = Linear::new(d_embed, d_embed, out_proj_bias, device);
        let d_head = d_embed / n_heads;
        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            n_heads,
            d_head,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>, y: &Tensor<B, f32>) -> Tensor<B, f32> {
        //     # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        //     # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        //     input_shape = x.shape
        let input_shape = x.shape().to_vec();

        //     batch_size, sequence_length, d_embed = input_shape
        let batch_size = x.size(0);

        //     # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        //     interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        let interim_shape = vec![
            batch_size as isize,
            -1,
            self.n_heads as isize,
            self.d_head as isize,
        ];

        //     # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        //     q = self.q_proj(x)
        let q = self.q_proj.forward(&x);
        //     # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        //     k = self.k_proj(y)
        let k = self.k_proj.forward(&y);
        //     # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        //     v = self.v_proj(y)
        let v = self.v_proj.forward(&y);

        //     # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        //     q = q.view(interim_shape).transpose(1, 2)
        let q = q.reshape(&interim_shape).transpose(1, 2);
        //     # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        //     k = k.view(interim_shape).transpose(1, 2)
        let k = k.reshape(&interim_shape).transpose(1, 2);
        //     # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        //     v = v.view(interim_shape).transpose(1, 2)
        let v = v.reshape(&interim_shape).transpose(1, 2);

        //     # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        //     weight = q @ k.transpose(-1, -2)
        let weight = q.matmul(&k).transpose(-1, -2);

        //     # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        //     weight /= math.sqrt(self.d_head)
        let weight = weight / (self.d_head as f64).sqrt();

        //     # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        //     weight = F.softmax(weight, dim=-1)
        let weight = weight.softmax(-1);

        //     # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        //     output = weight @ v
        let output = weight.matmul(&v);

        //     # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        //     output = output.transpose(1, 2).contiguous()
        let output = output.transpose(1, 2).contiguous();

        //     # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        //     output = output.view(input_shape)
        let output = output.reshape(input_shape);

        //     # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        //     output = self.out_proj(output)
        let output = self.out_proj.forward(&output);

        //     # (Batch_Size, Seq_Len_Q, Dim_Q)
        //     return output
        output
    }
}

impl<B: Backend> Layer<B, f32> for CrossAttention<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("q_proj".to_string(), &self.q_proj);
        map.insert("k_proj".to_string(), &self.k_proj);
        map.insert("v_proj".to_string(), &self.v_proj);
        map.insert("out_proj".to_string(), &self.out_proj);
        map
    }
}
