use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend,
    device::{self, Device},
    tensor::Tensor,
};
use rust_dnn_nn::{
    embedding::Embedding,
    layer::{Layer, LayerNorm, Linear},
    layer_list::LayerList,
};

use crate::self_attention::SelfAttention;

pub struct CLIPEmbedding<B: Backend> {
    token_embedding: Embedding<B, f32>,
    position_embedding: Tensor<B, f32>,
}

impl<B: Backend> CLIPEmbedding<B> {
    pub fn new(n_vocab: usize, n_embd: usize, n_token: usize, device: Device<B>) -> Self {
        let token_embedding = Embedding::new(n_vocab, n_embd, device).unwrap();
        let position_embedding = Tensor::zeros(vec![n_token, n_embd], device);
        Self {
            token_embedding,
            position_embedding,
        }
    }

    pub fn forward(&self, tokens: &Tensor<B, u32>) -> Tensor<B, f32> {
        let x = self.token_embedding.forward(&tokens);
        let x = x + &self.position_embedding;
        x
    }
}

impl<B: Backend> Layer<B, f32> for CLIPEmbedding<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}

pub struct CLIPLayer<B: Backend> {
    layernorm_1: LayerNorm<B, f32>,
    attention: SelfAttention<B>,
    layernorm_2: LayerNorm<B, f32>,
    linear_1: Linear<B, f32>,
    linear_2: Linear<B, f32>,
}

impl<B: Backend> CLIPLayer<B> {
    pub fn new(n_head: usize, n_embd: usize, device: Device<B>) -> Self {
        let layernorm_1 = LayerNorm::new(vec![n_embd], 1e-5, true, device);
        let attention = SelfAttention::new(n_head, n_embd, true, true, device);
        let layernorm_2 = LayerNorm::new(vec![n_embd], 1e-5, true, device);
        let linear_1 = Linear::new(n_embd, 4 * n_embd, true, device);
        let linear_2 = Linear::new(4 * n_embd, n_embd, true, device);
        Self {
            layernorm_1,
            attention,
            layernorm_2,
            linear_1,
            linear_2,
        }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        //         # (Batch_Size, Seq_Len, Dim)
        //         residue = x
        let residue = x.clone();

        //         ### SELF ATTENTION ###

        //         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x = self.layernorm_1(x)
        let x = self.layernorm_1.forward(&x);

        //         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x = self.attention(x, causal_mask=True)
        let x = self.attention.forward(&x, true);

        //         # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x += residue
        let x = x + residue;

        //         ### FEEDFORWARD LAYER ###
        //         # Apply a feedforward layer where the hidden dimension is 4 times the embedding dimension.

        //         residue = x
        let residue = x.clone();

        //         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x = self.layernorm_2(x)
        let x = self.layernorm_2.forward(&x);

        //         # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        //         x = self.linear_1(x)
        let x = self.linear_1.forward(&x);

        //         # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, 4 * Dim)
        //         x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
        let x = &x * (1.702 * &x).sigmoid();

        //         # (Batch_Size, Seq_Len, 4 * Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x = self.linear_2(x)
        let x = self.linear_2.forward(&x);

        //         # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        //         x += residue
        let x = x + residue;

        //         return x
        x
    }
}

impl<B: Backend> Layer<B, f32> for CLIPLayer<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}

pub struct CLIP<B: Backend> {
    embedding: CLIPEmbedding<B>,
    layers: LayerList<CLIPLayer<B>, B, f32>,
    layernorm: LayerNorm<B, f32>,
}

impl<B: Backend> CLIP<B> {
    pub fn new(device: Device<B>) -> Self {
        let embedding = CLIPEmbedding::new(49408, 768, 77, device);
        let mut layers = LayerList::new();
        for _ in 0..12 {
            layers.add(CLIPLayer::new(12, 768, device));
        }
        let layernorm = LayerNorm::new(vec![768], 1e-5, true, device);
        Self {
            embedding,
            layers,
            layernorm,
        }
    }

    pub fn forward(&self, tokens: &Tensor<B, u32>) -> Tensor<B, f32> {
        let mut state = self.embedding.forward(tokens);
        for layer in self.layers.layers() {
            state = layer.forward(&state);
        }
        let output = self.layernorm.forward(&state);
        output
    }
}

impl<B: Backend> Layer<B, f32> for CLIP<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        todo!()
    }
}
