// class Attention(nn.Module):
//     def __init__(self, args: ModelArgs):
//         super().__init__()
//         self.args = args
//         self.n_kv_heads = args.n_kv_heads
//         self.n_local_heads = args.n_heads
//         self.n_rep = self.n_local_heads // self.n_kv_heads
//         self.head_dim = args.dim // args.n_heads

//         self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
//         self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
//         self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
//         self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

//         self.cache_k = None
//         self.cache_v = None

//     def forward(self, x, start_pos, freqs_cis, mask):
//         bsz, seqlen, _ = x.shape
//         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

//         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
//         xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
//         xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

//         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

//         if self.cache_k is None or self.cache_k.shape[0] != bsz:
//             self.cache_k = torch.zeros((bsz, self.args.max_seq_len, self.n_kv_heads, self.head_dim)).to(xq)
//             self.cache_v = torch.zeros((bsz, self.args.max_seq_len, self.n_kv_heads, self.head_dim)).to(xq)

//         self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
//         self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

//         keys = self.cache_k[:bsz, : start_pos + seqlen]
//         values = self.cache_v[:bsz, : start_pos + seqlen]

//         keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_rep)
//         values = torch.repeat_interleave(values, dim=2, repeats=self.n_rep)

//         xq = xq.transpose(1, 2)
//         keys = keys.transpose(1, 2)
//         values = values.transpose(1, 2)

//         scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
//         if mask is not None:
//             scores = scores + mask
//         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
//         output = torch.matmul(scores, values)
//         return self.wo(output.transpose(1, 2).contiguous().view(bsz, seqlen, -1))

use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, tensor::Tensor,
};
use rust_dnn_nn::{
    function::scaled_dot_product_attention,
    layer::{Layer, Linear},
};

// class Attention(nn.Module):
//     def __init__(self, args: ModelArgs):
//         super().__init__()
//         self.args = args
//         self.n_heads = args.n_heads
//         self.head_dim = args.dim // args.n_heads

//         self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
//         self.wk = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
//         self.wv = nn.Linear(args.dim, self.n_heads * self.head_dim, bias=False)
//         self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)

//         self.cache_k = None
//         self.cache_v = None

//     def forward(self, x, start_pos, freqs_cis, mask):
//         bsz, seqlen, _ = x.shape

//         xq = self.wq(x)
//         xk = self.wk(x)
//         xv = self.wv(x)

//         xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
//         xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
//         xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

//         # Rotary Embedding
//         xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

//         # KV cache
//         if self.cache_k is None or self.cache_k.shape[0] != bsz:
//             self.cache_k = torch.zeros(
//                 (bsz, self.args.max_seq_len, self.n_heads, self.head_dim),
//                 device=x.device,
//                 dtype=x.dtype,
//             )
//             self.cache_v = torch.zeros(
//                 (bsz, self.args.max_seq_len, self.n_heads, self.head_dim),
//                 device=x.device,
//                 dtype=x.dtype,
//             )

//         self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
//         self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

//         keys = self.cache_k[:bsz, :start_pos + seqlen]
//         values = self.cache_v[:bsz, :start_pos + seqlen]

//         # (bsz, heads, seq, dim)
//         xq = xq.transpose(1, 2)
//         keys = keys.transpose(1, 2)
//         values = values.transpose(1, 2)

//         scores = torch.matmul(xq, keys.transpose(2, 3)) / (self.head_dim ** 0.5)
//         if mask is not None:
//             scores = scores + mask

//         scores = F.softmax(scores.float(), dim=-1).type_as(xq)
//         output = torch.matmul(scores, values)

//         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
//         return self.wo(output)

pub struct Attention<B: Backend, T: Float> {
    q_proj: Linear<B, T>,
    k_proj: Linear<B, T>,
    v_proj: Linear<B, T>,
    out_proj: Linear<B, T>,
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    dropout_ratio: f64,
    n_rep: usize,
    n_kv_heads: usize,
}

impl<B: Backend, T: Float> Attention<B, T> {
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        n_kv_heads: usize,
        dropout_ratio: f64,
        use_bias: bool,
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
            embed_dim,
            num_heads,
            n_kv_heads,
            head_dim,
            dropout_ratio,
            n_rep,
        }
    }

    pub fn forward(
        &self,
        q: &Tensor<B, T>,
        k: &Tensor<B, T>,
        v: &Tensor<B, T>,
        attn_mask: Option<&Tensor<B, T>>,
    ) -> Tensor<B, T> {
        let b = q.shape()[0];
        let t = q.shape()[1];
        let c = q.shape()[2];

        let q = q.reshape(vec![b * t, c]);
        let q = self.q_proj.forward(&q);
        let q = q.reshape(vec![b, t, c]);

        let k = k.reshape(vec![b * t, c]);
        let k = self.k_proj.forward(&k);
        let k = k.reshape(vec![b, t, self.n_kv_heads * self.head_dim]);

        let v = v.reshape(vec![b * t, c]);
        let v = self.v_proj.forward(&v);
        let v = v.reshape(vec![b, t, self.n_kv_heads * self.head_dim]);

        let k = k.repeat_interleave(2, self.n_rep);
        let v = v.repeat_interleave(2, self.n_rep);

        let q = q
            .reshape(vec![b, t, self.num_heads, self.head_dim])
            .permuted_axes(&[0, 2, 1, 3]);
        let k = k
            .reshape(vec![b, t, self.num_heads, self.head_dim])
            .permuted_axes(&[0, 2, 1, 3]);
        let v = v
            .reshape(vec![b, t, self.num_heads, self.head_dim])
            .permuted_axes(&[0, 2, 1, 3]);

        let attn_output =
            scaled_dot_product_attention(&q, &k, &v, attn_mask, self.dropout_ratio, true, None);
        let attn_output = attn_output
            .permuted_axes(&[0, 2, 1, 3])
            .reshape(vec![b, t, c]);

        let attn_output = attn_output.reshape(vec![b * t, c]);
        let y = self.out_proj.forward(&attn_output);
        y.reshape(vec![b, t, c])
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
        map.insert("q_proj".to_string(), &self.q_proj);
        map.insert("k_proj".to_string(), &self.k_proj);
        map.insert("v_proj".to_string(), &self.v_proj);
        map.insert("out_proj".to_string(), &self.out_proj);
        map
    }
}
