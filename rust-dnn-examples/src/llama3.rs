use std::collections::HashMap;

use rust_dnn_core::{backend::Backend, device::Device, float::Float, tensor::Tensor};
use rust_dnn_nn::{
    embedding::Embedding,
    function::scaled_dot_product_attention,
    layer::{Layer, Linear, RMSNorm},
    layer_list::LayerList,
    sequential::SequentialItem,
};

pub fn precompute_freqs_cis<B: Backend, T: Float>(
    dim: usize,
    end: usize,
    theta: f64,
    device: Device<B>,
) -> Tensor<B, T> {
    let theta = Tensor::<B, T>::from_f64(theta, device);
    let theta_pow_rhs = Tensor::arange_step(
        T::from_usize(0),
        T::from_usize(dim),
        T::from_usize(2),
        device,
    )
    .narrow(0, 0, dim / 2)
        / dim as f64;

    let freqs = 1.0 / (theta.pow(&theta_pow_rhs));
    let t = Tensor::arange(0..(end as isize), device);
    let freqs = t.unsqueeze(1) * freqs.unsqueeze(0);
    let cos = freqs.cos();
    let sin = freqs.sin();
    Tensor::stack(&[cos, sin], -1)
}

fn apply_rotary_emb<B: Backend, T: Float>(
    xq: Tensor<B, T>,
    xk: Tensor<B, T>,
    freqs_cis: &Tensor<B, T>,
) -> (Tensor<B, T>, Tensor<B, T>) {
    let mut xq_real_shape = xq.shape().to_vec();
    xq_real_shape.pop();
    xq_real_shape.push(xq.size(-1) / 2);
    xq_real_shape.push(2);
    let xq_real = xq.reshape(xq_real_shape);

    let mut xk_real_shape = xk.shape().to_vec();
    xk_real_shape.pop();
    xk_real_shape.push(xq.size(-1) / 2);
    xk_real_shape.push(2);
    let xk_real = xk.reshape(xk_real_shape);

    let cos = freqs_cis
        .select(-1, 0)
        .reshape(vec![1, xq_real.size(1), 1, xq_real.size(3)]);
    let sin = freqs_cis
        .select(-1, 1)
        .reshape(vec![1, xq_real.size(1), 1, xq_real.size(3)]);

    let xq_out_r = xq_real.select(-1, 0) * &cos - xq_real.select(-1, 1) * &sin;
    let xq_out_i = xq_real.select(-1, 0) * &sin + xq_real.select(-1, 1) * &cos;

    let xk_out_r = xk_real.select(-1, 0) * &cos - xk_real.select(-1, 1) * &sin;
    let xk_out_i = xk_real.select(-1, 0) * &sin + xk_real.select(-1, 1) * &cos;

    let xq_out = Tensor::stack(&[xq_out_r, xq_out_i], -1).flatten(3, -1);
    let xk_out = Tensor::stack(&[xk_out_r, xk_out_i], -1).flatten(3, -1);

    (xq_out, xk_out)
}

pub struct Attention<B: Backend, T: Float> {
    q_proj: Linear<B, T>,
    k_proj: Linear<B, T>,
    v_proj: Linear<B, T>,
    out_proj: Linear<B, T>,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    n_rep: usize,
    n_kv_heads: usize,
    use_kvcache: bool,
    cache_k: Option<Tensor<B, T>>,
    cache_v: Option<Tensor<B, T>>,
}

impl<B: Backend, T: Float> Attention<B, T> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        use_bias: bool,
        use_kvcache: bool,
        device: Device<B>,
    ) -> Self {
        let head_dim = embed_dim / num_heads;
        let q_proj = Linear::new(embed_dim, embed_dim, use_bias, device);
        let k_proj = Linear::new(embed_dim, n_kv_heads * head_dim, use_bias, device);
        let v_proj = Linear::new(embed_dim, n_kv_heads * head_dim, use_bias, device);
        let out_proj = Linear::new(embed_dim, embed_dim, use_bias, device);

        let n_rep = num_heads / n_kv_heads;

        Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            n_kv_heads,
            max_seq_len,
            head_dim,
            n_rep,
            use_kvcache,
            cache_k: None,
            cache_v: None,
        }
    }

    pub fn forward(
        &mut self,
        q: &Tensor<B, T>,
        k: &Tensor<B, T>,
        v: &Tensor<B, T>,
        start_pos: usize,
        freqs_cis: &Tensor<B, T>,
        attn_mask: Option<&Tensor<B, T>>,
        is_train: bool,
    ) -> Tensor<B, T> {
        let b = q.shape()[0];
        let t = q.shape()[1];
        let c = q.shape()[2];

        let q = q.reshape(vec![b * t, c]);
        let q = self.q_proj.forward(&q);
        let q = q.reshape(vec![b, t, self.num_heads, self.head_dim]);

        let k = k.reshape(vec![b * t, c]);
        let k = self.k_proj.forward(&k);
        let k = k.reshape(vec![b, t, self.n_kv_heads, self.head_dim]);

        let v = v.reshape(vec![b * t, c]);
        let v = self.v_proj.forward(&v);
        let v = v.reshape(vec![b, t, self.n_kv_heads, self.head_dim]);

        let (q, k) = apply_rotary_emb(q, k, freqs_cis);

        let (k, v) = if self.use_kvcache && !is_train {
            self.update_kvcache(k, v, start_pos, t)
        } else {
            (k, v)
        };

        let k = k.repeat_interleave(2, self.n_rep);
        let v = v.repeat_interleave(2, self.n_rep);

        let q = q.permuted_axes(&[0, 2, 1, 3]);
        let k = k.permuted_axes(&[0, 2, 1, 3]);
        let v = v.permuted_axes(&[0, 2, 1, 3]);

        let attn_output = scaled_dot_product_attention(&q, &k, &v, attn_mask, 0.0, true, None);
        let attn_output = attn_output
            .permuted_axes(&[0, 2, 1, 3])
            .reshape(vec![b, t, c]);

        let attn_output = attn_output.reshape(vec![b * t, c]);
        let y = self.out_proj.forward(&attn_output);
        y.reshape(vec![b, t, c])
    }

    fn update_kvcache(
        &mut self,
        k: Tensor<B, T>,
        v: Tensor<B, T>,
        start_pos: usize,
        t: usize,
    ) -> (Tensor<B, T>, Tensor<B, T>) {
        let b = k.shape()[0];

        if self.cache_k.is_none() || self.cache_k.as_ref().unwrap().size(0) != b {
            let cache_k = Tensor::zeros(
                vec![b, self.max_seq_len, self.n_kv_heads, self.head_dim],
                k.device(),
            );
            self.cache_k = Some(cache_k);

            let cache_v = Tensor::zeros(
                vec![b, self.max_seq_len, self.n_kv_heads, self.head_dim],
                k.device(),
            );
            self.cache_v = Some(cache_v);
        }

        let cache_k = self.cache_k.as_mut().unwrap();
        cache_k.set_item(
            &vec![
                (0, b),
                (start_pos, start_pos + t),
                (0, cache_k.size(2)),
                (0, cache_k.size(3)),
            ],
            &k,
        );
        let cache_v = self.cache_v.as_mut().unwrap();
        cache_v.set_item(
            &vec![
                (0, b),
                (start_pos, start_pos + t),
                (0, cache_k.size(2)),
                (0, cache_k.size(3)),
            ],
            &v,
        );

        let k = cache_k.get_item(vec![
            (0, b),
            (0, start_pos + t),
            (0, cache_k.size(2)),
            (0, cache_k.size(3)),
        ]);
        let v = cache_v.get_item(vec![
            (0, b),
            (0, start_pos + t),
            (0, cache_k.size(2)),
            (0, cache_k.size(3)),
        ]);

        (k, v)
    }

    pub fn q_proj(&self) -> &Linear<B, T> {
        &self.q_proj
    }

    pub fn k_proj(&self) -> &Linear<B, T> {
        &self.k_proj
    }

    pub fn v_proj(&self) -> &Linear<B, T> {
        &self.v_proj
    }

    pub fn out_proj(&self) -> &Linear<B, T> {
        &self.out_proj
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Attention<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("wq".to_string(), &self.q_proj);
        map.insert("wk".to_string(), &self.k_proj);
        map.insert("wv".to_string(), &self.v_proj);
        map.insert("wo".to_string(), &self.out_proj);
        map
    }

    fn parameters_map(&self) -> std::collections::HashMap<String, Tensor<B, T>> {
        std::collections::HashMap::new()
    }

    fn trainable_parameters_map(&self) -> std::collections::HashMap<String, Tensor<B, T>> {
        self.parameters_map()
    }

    fn all_parameters_map(&self) -> std::collections::HashMap<String, Tensor<B, T>> {
        let mut map = std::collections::HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_parameters_map() {
                let name = std::format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn all_trainable_parameters_map(&self) -> std::collections::HashMap<String, Tensor<B, T>> {
        let mut map = std::collections::HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_trainable_parameters_map() {
                let name = std::format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.trainable_parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn load_parameters_map(
        &mut self,
        map: std::collections::HashMap<String, Tensor<B, T>>,
    ) -> rust_dnn_core::error::Result<()> {
        for (name, parameter) in self.all_parameters_map() {
            if let Some(param) = map.get(&name) {
                parameter.copy(param);
            }
        }
        Ok(())
    }
}

struct FeedForward<B: Backend> {
    w1: Linear<B, f32>,
    w2: Linear<B, f32>,
    w3: Linear<B, f32>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(n_embd: usize, hidden_dim: usize, device: Device<B>) -> Self {
        let w1 = Linear::new(n_embd, hidden_dim, false, device);
        let w2 = Linear::new(hidden_dim, n_embd, false, device);
        let w3 = Linear::new(n_embd, hidden_dim, false, device);
        Self { w1, w2, w3 }
    }
}

impl<B: Backend> SequentialItem<B, f32> for FeedForward<B> {
    fn forward(&mut self, x: Tensor<B, f32>, is_train: bool) -> Tensor<B, f32> {
        self.w2
            .forward(&(self.w1.forward(&x).silu() * self.w3.forward(&x)))
    }
}

impl<B: Backend> Layer<B, f32> for FeedForward<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("w1".to_string(), &self.w1);
        map.insert("w2".to_string(), &self.w2);
        map.insert("w3".to_string(), &self.w3);
        map
    }
}

struct TransformerBlock<B: Backend> {
    attention_norm: RMSNorm<B, f32>,
    attention: Attention<B, f32>,
    ffn_norm: RMSNorm<B, f32>,
    feed_forward: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        n_embd: usize,
        n_head: usize,
        hidden_size: usize,
        max_seq_len: usize,
        use_bias: bool,
        use_kvcache: bool,
        device: Device<B>,
    ) -> Self {
        let attention_norm = RMSNorm::new(n_embd, 1e-5, device);
        let attention = Attention::new(
            n_embd,
            n_head,
            n_head / 4,
            max_seq_len,
            use_bias,
            use_kvcache,
            device,
        );
        let ffn_norm = RMSNorm::new(n_embd, 1e-5, device);
        let feed_forward = FeedForward::new(n_embd, hidden_size, device);
        Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        }
    }

    pub fn forward(
        &mut self,
        x: Tensor<B, f32>,
        start_pos: usize,
        freqs_cis: &Tensor<B, f32>,
        mask: Option<&Tensor<B, f32>>,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let h = self.attention_norm.forward(&x);
        let x = &x
            + self
                .attention
                .forward(&h, &h, &h, start_pos, freqs_cis, mask, is_train);
        let h = self.ffn_norm.forward(&x);
        let x = &x + self.feed_forward.forward(h, is_train);
        x
    }
}

impl<B: Backend> Layer<B, f32> for TransformerBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("attention_norm".to_string(), &self.attention_norm);
        map.insert("attention".to_string(), &self.attention);
        map.insert("ffn_norm".to_string(), &self.ffn_norm);
        map.insert("feed_forward".to_string(), &self.feed_forward);
        map
    }
}

pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B, f32>,
    layers: LayerList<TransformerBlock<B>, B, f32>,
    norm: RMSNorm<B, f32>,
    output: Linear<B, f32>,
    freqs_cis: Tensor<B, f32>,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layers: usize,
        n_embd: usize,
        n_head: usize,
        hidden_size: usize,
        use_bias: bool,
        use_kvcache: bool,
        device: Device<B>,
    ) -> Self {
        let tok_embeddings = Embedding::new(vocab_size, n_embd, device).unwrap();
        let blocks = (0..n_layers)
            .into_iter()
            .map(|_| {
                TransformerBlock::new(
                    n_embd,
                    n_head,
                    hidden_size,
                    block_size,
                    use_bias,
                    use_kvcache,
                    device,
                )
            })
            .collect();
        let layers = LayerList::from_vec(blocks);
        let norm = RMSNorm::new(n_embd, 1e-5, device);
        let lm_head = Linear::new(n_embd, vocab_size, false, device);
        let freqs_cis = precompute_freqs_cis(n_embd / n_head, block_size * 2, 500000.0, device);

        Self {
            tok_embeddings,
            layers,
            norm,
            output: lm_head,
            freqs_cis,
        }
    }

    pub fn forward(
        &mut self,
        idx: &Tensor<B, u32>,
        start_pos: usize,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let mut x = self.tok_embeddings.forward(idx);
        let seqlen = idx.size(1);
        let freqs_cis = self.freqs_cis.narrow(0, start_pos, seqlen);

        let ones = Tensor::fill(vec![seqlen, seqlen], -f32::MAX, x.device());
        let mask = ones.triu2(1);
        let mask = if seqlen > 1 { Some(&mask) } else { None };

        for layer in self.layers.layers_mut() {
            x = layer.forward(x, start_pos, &freqs_cis, mask, is_train);
        }

        x = self.norm.forward(&x);
        if !is_train {
            x = x.get_item(vec![
                (0, x.size(0)),
                (x.size(1) - 1, x.size(1)),
                (0, x.size(2)),
            ]);
        }
        self.output.forward(&x)
    }
}

impl<B: Backend> Layer<B, f32> for Transformer<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("tok_embeddings".to_string(), &self.tok_embeddings);
        map.insert("layers".to_string(), &self.layers);
        map.insert("norm".to_string(), &self.norm);
        map.insert("output".to_string(), &self.output);
        map
    }
}
