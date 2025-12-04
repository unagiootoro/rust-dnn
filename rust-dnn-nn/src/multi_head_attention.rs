use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, tensor::Tensor,
};

use crate::{
    function::scaled_dot_product_attention,
    layer::{Layer, Linear},
};

pub struct MultiHeadAttention<B: Backend, T: Float> {
    q_proj: Linear<B, T>,
    k_proj: Linear<B, T>,
    v_proj: Linear<B, T>,
    out_proj: Linear<B, T>,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    dropout_ratio: f64,
}

impl<B: Backend, T: Float> MultiHeadAttention<B, T> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        dropout_ratio: f64,
        use_bias: bool,
        device: Device<B>,
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let q_proj = Linear::new(embed_dim, embed_dim, use_bias, device)?;
        let k_proj = Linear::new(embed_dim, embed_dim, use_bias, device)?;
        let v_proj = Linear::new(embed_dim, embed_dim, use_bias, device)?;
        let out_proj = Linear::new(embed_dim, embed_dim, use_bias, device)?;
        let multi_head_attention = Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            embed_dim,
            num_heads,
            head_dim,
            dropout_ratio,
        };
        Ok(multi_head_attention)
    }

    pub fn forward(
        &self,
        q: &Tensor<B, T>,
        k: &Tensor<B, T>,
        v: &Tensor<B, T>,
        attn_mask: Option<&Tensor<B, T>>,
    ) -> Result<Tensor<B, T>> {
        let b = q.shape()[0];
        let t = q.shape()[1];
        let c = q.shape()[2];

        let q = q.reshape(vec![b * t, c])?;
        let q = self.q_proj.forward(&q)?;
        let q = q.reshape(vec![b, t, c])?;
        let k = k.reshape(vec![b * t, c])?;
        let k = self.k_proj.forward(&k)?;
        let k = k.reshape(vec![b, t, c])?;
        let v = v.reshape(vec![b * t, c])?;
        let v = self.v_proj.forward(&v)?;
        let v = v.reshape(vec![b, t, c])?;

        let q = q
            .reshape(vec![b, t, self.num_heads, self.head_dim])?
            .permuted_axes(&[0, 2, 1, 3])?;
        let k = k
            .reshape(vec![b, t, self.num_heads, self.head_dim])?
            .permuted_axes(&[0, 2, 1, 3])?;
        let v = v
            .reshape(vec![b, t, self.num_heads, self.head_dim])?
            .permuted_axes(&[0, 2, 1, 3])?;

        let attn_output =
            scaled_dot_product_attention(&q, &k, &v, attn_mask, self.dropout_ratio, true, None)?;
        let attn_output = attn_output
            .permuted_axes(&[0, 2, 1, 3])?
            .reshape(vec![b, t, c])?;

        let attn_output = attn_output.reshape(vec![b * t, c])?;
        let y = self.out_proj.forward(&attn_output)?;
        let y = y.reshape(vec![b, t, c])?;
        Ok(y)
    }
}

impl<B: Backend, T: Float> Layer<B, T> for MultiHeadAttention<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("q_proj".to_string(), &self.q_proj);
        map.insert("k_proj".to_string(), &self.k_proj);
        map.insert("v_proj".to_string(), &self.v_proj);
        map.insert("out_proj".to_string(), &self.out_proj);
        map
    }
}
