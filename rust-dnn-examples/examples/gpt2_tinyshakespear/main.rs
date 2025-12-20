use std::{collections::HashMap, fs};

use rust_dnn_core::{
    backend::Backend,
    cpu_backend::CpuBackend,
    device::{self, Device},
    float::Float,
    num::Num,
    tensor::Tensor,
};
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::{Batchable, batch_iter},
    embedding::Embedding,
    function::generate_causal_attention_mask,
    layer::{Layer, LayerNorm, Linear},
    loss::*,
    multi_head_attention::MultiHeadAttention,
    optimizer::{Adam, Optimizer},
    sequential::{Sequential, SequentialItem},
};
use rust_dnn_safetensors::{deserialize, serialize};

struct CharTokenizer {
    vocab_list: Vec<char>,
}

impl CharTokenizer {
    pub fn new() -> Self {
        Self {
            vocab_list: Vec::new(),
        }
    }

    pub fn encode(&self, chr: char) -> Option<usize> {
        self.vocab_list.iter().position(|c| *c == chr)
    }

    pub fn encode_and_register_token(&mut self, chr: char) -> usize {
        let pos = self.vocab_list.iter().position(|c| *c == chr);
        if let Some(pos) = pos {
            pos
        } else {
            let pos = self.vocab_list.len();
            self.vocab_list.push(chr);
            pos
        }
    }

    pub fn decode(&self, idx: usize) -> Option<char> {
        let chr = self.vocab_list.get(idx);
        if let Some(chr) = chr {
            Some(*chr)
        } else {
            None
        }
    }
}

struct MLP<B: Backend> {
    fc: Linear<B, f32>,
    proj: Linear<B, f32>,
    dropout_ratio: f64,
}

impl<B: Backend> MLP<B> {
    pub fn new(n_embd: usize, dropout_ratio: f64, use_bias: bool, device: Device<B>) -> Self {
        let fc = Linear::new(n_embd, 4 * n_embd, use_bias, device);
        let proj = Linear::new(4 * n_embd, n_embd, use_bias, device);
        Self {
            fc,
            proj,
            dropout_ratio,
        }
    }
}

impl<B: Backend> SequentialItem<B, f32> for MLP<B> {
    fn forward(&mut self, x: Tensor<B, f32>, is_train: bool) -> Tensor<B, f32> {
        let mut x = self.fc.forward(&x);
        x = x.gelu();
        x = self.proj.forward(&x);
        if self.dropout_ratio > 0.0 {
            x = x.dropout(self.dropout_ratio, is_train, None);
        }
        x
    }
}

impl<B: Backend> Layer<B, f32> for MLP<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("fc".to_string(), &self.fc);
        map.insert("proj".to_string(), &self.proj);
        map
    }
}

struct Block<B: Backend> {
    ln1: LayerNorm<B, f32>,
    attn: MultiHeadAttention<B, f32>,
    ln2: LayerNorm<B, f32>,
    mlp: MLP<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(
        n_embd: usize,
        n_head: usize,
        dropout_ratio: f64,
        use_bias: bool,
        device: Device<B>,
    ) -> Self {
        let ln1 = LayerNorm::new(vec![n_embd], 1e-5, use_bias, device);
        let attn = MultiHeadAttention::new(n_embd, n_head, dropout_ratio, use_bias, device);
        let ln2 = LayerNorm::new(vec![n_embd], 1e-5, use_bias, device);
        let mlp = MLP::new(n_embd, dropout_ratio, use_bias, device);
        Self {
            ln1,
            attn,
            ln2,
            mlp,
        }
    }
}

impl<B: Backend> SequentialItem<B, f32> for Block<B> {
    fn forward(&mut self, x: Tensor<B, f32>, is_train: bool) -> Tensor<B, f32> {
        let h = self.ln1.forward(&x);
        let mask = generate_causal_attention_mask(x.size(1), h.device()).unwrap();
        let x = &x + self.attn.forward(&h, &h, &h, Some(&mask));
        let h = self.ln2.forward(&x);
        let x = &x + self.mlp.forward(h, is_train);
        x
    }
}

impl<B: Backend> Layer<B, f32> for Block<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("ln1".to_string(), &self.ln1);
        map.insert("attn".to_string(), &self.attn);
        map.insert("ln2".to_string(), &self.ln2);
        map.insert("mlp".to_string(), &self.mlp);
        map
    }
}

struct GPT<B: Backend> {
    wte: Embedding<B, f32>,
    wpe: Embedding<B, f32>,
    h: Sequential<Block<B>, B, f32>,
    ln_f: LayerNorm<B, f32>,
    lm_head: Linear<B, f32>,
    dropout_ratio: f64,
}

impl<B: Backend> GPT<B> {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layers: usize,
        n_embd: usize,
        n_head: usize,
        dropout_ratio: f64,
        use_bias: bool,
        device: Device<B>,
    ) -> Self {
        let wte = Embedding::new(vocab_size, n_embd, device).unwrap();
        let wpe = Embedding::new(block_size, n_embd, device).unwrap();
        let blocks = (0..n_layers)
            .into_iter()
            .map(|_| Block::new(n_embd, n_head, dropout_ratio, use_bias, device))
            .collect();
        let h = Sequential::from_vec(blocks);
        let ln_f = LayerNorm::new(vec![n_embd], 1e-5, use_bias, device);
        let lm_head = Linear::new(n_embd, vocab_size, false, device);
        // wte.weight = Rc::clone(&lm_head.weight); // TODO: 暫定無効
        Self {
            wte,
            wpe,
            h,
            ln_f,
            lm_head,
            dropout_ratio,
        }
    }

    pub fn forward(&mut self, idx: &Tensor<B, u32>, is_train: bool) -> Tensor<B, f32> {
        let pos =
            Tensor::arange(0..(idx.size(1) as isize), idx.device()).reshape(vec![1, idx.size(1)]);

        let tok_emb = self.wte.forward(idx);
        let pos_emb = self.wpe.forward(&pos);
        let mut x = tok_emb + pos_emb;
        if self.dropout_ratio > 0.0 {
            x = x.dropout(self.dropout_ratio, is_train, None);
        }
        let mut x = self.h.forward(x.clone(), is_train);
        x = self.ln_f.forward(&x);
        if !is_train {
            x = x.get_item(vec![
                (0, x.size(0)),
                (x.size(1) - 1, x.size(1)),
                (0, x.size(2)),
            ]);
        }
        self.lm_head.forward(&x)
    }
}

impl<B: Backend> Layer<B, f32> for GPT<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("wte".to_string(), &self.wte);
        map.insert("wpe".to_string(), &self.wpe);
        map.insert("h".to_string(), &self.h);
        map.insert("ln_f".to_string(), &self.ln_f);
        map.insert("lm_head".to_string(), &self.lm_head);
        map
    }
}

// fn load_weights<B: Backend>(
//     model: &mut GPT<B>,
//     file_path: &str,
//     n_embd: usize,
//     n_layers: usize,
// ) {
//     println!("begin load file");
//     let model_safetensors = fs::read(file_path).unwrap();
//     println!("end load file");
//     println!("begin deserialize safetensors");
//     let params = safetensors::deserialize(model_safetensors);
//     println!("end deserialize safetensors");

//     println!("begin load parameter");
//     model
//         .lm_head
//         .weight
//         .borrow_mut()
//         .copy(&params["lm_head.weight"]);
//     model.wte.weight = Rc::clone(&model.lm_head.weight);

//     for layer_index in 0..n_layers {
//         let c_attn_weight = &params[&format!("transformer.h.{}.attn.c_attn.weight", layer_index)];
//         let c_attn_weight_vec = c_attn_weight.split(0, vec![n_embd, n_embd, n_embd]);
//         model
//             .h
//             .get_mut(layer_index)
//             .attn
//             .q_proj
//             .weight
//             .borrow_mut()
//             .copy(&c_attn_weight_vec[0]);
//         model
//             .h
//             .get_mut(layer_index)
//             .attn
//             .k_proj
//             .weight
//             .borrow_mut()
//             .copy(&c_attn_weight_vec[1]);
//         model
//             .h
//             .get_mut(layer_index)
//             .attn
//             .v_proj
//             .weight
//             .borrow_mut()
//             .copy(&c_attn_weight_vec[2]);

//         let c_attn_bias = &params[&format!("transformer.h.{}.attn.c_attn.bias", layer_index)];
//         let c_attn_bias_vec = c_attn_bias.split(0, vec![n_embd, n_embd, n_embd]);
//         if let Some(ref bias) = model.h.get_mut(layer_index).attn.q_proj.bias {
//             bias.borrow_mut().copy(&c_attn_bias_vec[0]);
//         }
//         if let Some(ref bias) = model.h.get_mut(layer_index).attn.k_proj.bias {
//             bias.borrow_mut().copy(&c_attn_bias_vec[1]);
//         }
//         if let Some(ref bias) = model.h.get_mut(layer_index).attn.v_proj.bias {
//             bias.borrow_mut().copy(&c_attn_bias_vec[2]);
//         }

//         model
//             .h
//             .get_mut(layer_index)
//             .attn
//             .out_proj
//             .weight
//             .borrow_mut()
//             .copy(&params[&format!("transformer.h.{}.attn.c_proj.weight", layer_index)]);
//         if let Some(ref bias) = model.h.get_mut(layer_index).attn.out_proj.bias {
//             bias.borrow_mut()
//                 .copy(&params[&format!("transformer.h.{}.attn.c_proj.bias", layer_index)]);
//         }

//         model
//             .h
//             .get_mut(layer_index)
//             .ln1
//             .gamma
//             .borrow_mut()
//             .copy(&params[&format!("transformer.h.{}.ln_1.weight", layer_index)]);
//         if let Some(ref beta) = model.h.get_mut(layer_index).ln1.beta {
//             beta.borrow_mut()
//                 .copy(&params[&format!("transformer.h.{}.ln_1.bias", layer_index)]);
//         }

//         model
//             .h
//             .get_mut(layer_index)
//             .ln2
//             .gamma
//             .borrow_mut()
//             .copy(&params[&format!("transformer.h.{}.ln_2.weight", layer_index)]);
//         if let Some(ref beta) = model.h.get_mut(layer_index).ln2.beta {
//             beta.borrow_mut()
//                 .copy(&params[&format!("transformer.h.{}.ln_2.bias", layer_index)]);
//         }

//         model
//             .h
//             .get_mut(layer_index)
//             .mlp
//             .fc
//             .weight
//             .borrow_mut()
//             .copy(&params[&format!("transformer.h.{}.mlp.c_fc.weight", layer_index)]);
//         if let Some(ref bias) = model.h.get_mut(layer_index).mlp.fc.bias {
//             bias.borrow_mut()
//                 .copy(&params[&format!("transformer.h.{}.mlp.c_fc.bias", layer_index)]);
//         }

//         model
//             .h
//             .get_mut(layer_index)
//             .mlp
//             .proj
//             .weight
//             .borrow_mut()
//             .copy(&params[&format!("transformer.h.{}.mlp.c_proj.weight", layer_index)]);
//         if let Some(ref bias) = model.h.get_mut(layer_index).mlp.proj.bias {
//             bias.borrow_mut()
//                 .copy(&params[&format!("transformer.h.{}.mlp.c_proj.bias", layer_index)]);
//         }
//     }

//     model
//         .ln_f
//         .gamma
//         .borrow_mut()
//         .copy(&params["transformer.ln_f.weight"]);
//     if let Some(ref beta) = model.ln_f.beta {
//         beta.borrow_mut().copy(&params["transformer.ln_f.bias"]);
//     }
//     println!("end load parameter");

//     model
//         .wpe
//         .weight
//         .borrow_mut()
//         .copy(&params["transformer.wpe.weight"]);
// }

fn generate<B: Backend>(
    gpt: &mut GPT<B>,
    mut idx: Tensor<B, u32>,
    max_new_tokens: usize,
    block_size: usize,
) -> Tensor<B, u32> {
    for iter in 0..max_new_tokens {
        println!("generate iter = {}", iter);
        let idx_cond = if idx.size(1) <= block_size {
            idx.clone()
        } else {
            let ranges = vec![(0, idx.size(0)), (idx.size(1) - block_size, idx.size(1))];
            idx.get_item(ranges)
        };
        let logits = gpt.forward(&idx_cond, false);
        let ranges = vec![
            (0, logits.size(0)),
            (logits.size(1) - 1, logits.size(1)),
            (0, logits.size(2)),
        ];
        let logits = logits.get_item(ranges);
        let probs = logits.softmax(2);
        let idx_next = probs.multinomial(1, None);
        let idx_next = idx_next.reshape(vec![1, 1]);
        idx = Tensor::cat(&vec![idx, idx_next], 1);
    }

    idx
}

// fn main() {
//     // Shakespeare
//     // let vocab_size = 65;
//     // let block_size = 64;
//     // let n_layers = 4;
//     // let n_embd = 128;
//     // let n_head = 4;

//     // let vocab_size = 65;
//     // let block_size = 256;
//     // let n_layers = 6;
//     // let n_embd = 384;
//     // let n_head = 6;

//     // GPT2
//     let vocab_size = 50257;
//     let block_size = 1024;

//     // GPT2
//     // let n_layers = 12;
//     // let n_embd = 768;
//     // let n_head = 12;

//     // GPT2-XL
//     let n_layers = 48;
//     let n_embd = 1600;
//     let n_head = 25;

//     let dropout_ratio = 0.0;
//     let mut gpt = GPT::new(
//         vocab_size,
//         block_size,
//         n_layers,
//         n_embd,
//         n_head,
//         dropout_ratio,
//         true,
//     );
//     load_weights(
//         &mut gpt,
//         "exclude/gpt2_xl_model.safetensors",
//         n_embd,
//         n_layers,
//     );
//     let input = Tensor::from_vec(
//         vec![1, 13],
//         vec![
//             2061.0, 318.0, 262.0, 3280.0, 284.0, 1204.0, 11.0, 262.0, 6881.0, 11.0, 290.0, 2279.0,
//             30.0,
//         ],
//     );

//     let result = generate(&mut gpt, input, 100, block_size);
//     println!("result.shape() = {:?}", result.shape());
//     let float_vec = result.to_vec();
//     let int_vec: Vec<i32> = float_vec.iter().map(|&x| x as i32).collect();
//     println!("result.to_vec() = {:?}", int_vec);
// }

// fn main() {
//     let vocab_size = 65;
//     let block_size = 64;
//     let n_layers = 4;
//     let n_embd = 128;
//     let n_head = 4;

//     let dropout_ratio = 0.0;
//     let mut gpt = GPT::new(
//         vocab_size,
//         block_size,
//         n_layers,
//         n_embd,
//         n_head,
//         dropout_ratio,
//         true,
//     );

//     let input = Tensor::from_vec(
//         vec![1, 3],
//         vec![0.0, 1.0, 2.0],
//     );

//     let result = generate(&mut gpt, input, 100, block_size);
//     println!("result.shape() = {:?}", result.shape());
//     let float_vec = result.to_vec();
//     let int_vec: Vec<i32> = float_vec.iter().map(|&x| x as i32).collect();
//     println!("result.to_vec() = {:?}", int_vec);
// }

pub struct CustomDataset {
    input_chars: Vec<char>,
    tokenizer: CharTokenizer,
}

impl CustomDataset {
    pub fn new(input: String) -> Self {
        let mut tokenizer = CharTokenizer::new();
        let input_chars = input.chars().collect::<Vec<char>>();
        for i in 0..input_chars.len() {
            tokenizer.encode_and_register_token(input_chars[i]);
        }
        Self {
            input_chars,
            tokenizer,
        }
    }

    pub fn len(&self) -> usize {
        32000
    }

    pub fn get(&self, index: usize) -> (Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>) {
        let block_size = 64;

        let mut idxs = Vec::new();
        let i = index;
        for j in i..(i + block_size) {
            let idx = self.tokenizer.encode(self.input_chars[j]).unwrap();
            idxs.push(idx as u32);
        }
        let x = Tensor::from_vec(idxs, vec![block_size], Device::get_cpu_device());

        let mut idxs2 = Vec::new();
        let i = index + 1;
        for j in i..(i + block_size) {
            let idx = self.tokenizer.encode(self.input_chars[j]).unwrap();
            idxs2.push(idx as u32);
        }
        let y = Tensor::from_vec(idxs2, vec![block_size], Device::get_cpu_device());

        (x.reshape(vec![1, x.len()]), y.reshape(vec![1, y.len()]))
    }
}

impl Batchable<(Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>)> for CustomDataset {
    fn get_batch(&self, index: &[u32]) -> (Tensor<CpuBackend, u32>, Tensor<CpuBackend, u32>) {
        let mut x_list = Vec::new();
        let mut y_list = Vec::new();
        for i in index {
            let (x, y) = self.get(*i as usize);
            x_list.push(x);
            y_list.push(y);
        }

        let xs = Tensor::cat(&x_list, 0);
        let ys = Tensor::cat(&y_list, 0);
        (xs, ys)
    }

    fn batch_size(&self) -> usize {
        self.len()
    }
}

fn run<B: Backend>(device: Device<B>) {
    let vocab_size = 65;
    let block_size = 64;
    let n_layers = 4;
    let n_embd = 128;
    let n_head = 4;

    let input = fs::read_to_string("../datasets/tinyshakespeare/input.txt").unwrap();
    let dataset = CustomDataset::new(input);
    println!(
        "dataset.tokenizer.vocab_list.len() = {}",
        dataset.tokenizer.vocab_list.len()
    );

    let dropout_ratio = 0.0;
    let mut gpt = GPT::<B>::new(
        vocab_size,
        block_size,
        n_layers,
        n_embd,
        n_head,
        dropout_ratio,
        true,
        device,
    );

    {
        // set_train(true);

        let mut optimizer = Adam::default();

        let batch_size = 64;
        let max_epochs = 1;

        // let train_loader = DataLoader::new(&dataset, batch_size, true, None);
        for epoch in 0..max_epochs {
            println!("epoch = {}", epoch);
            // for (iter, [intputs, outputs]) in train_loader.iter().enumerate() {
            for (iter, (intputs, outputs)) in
                batch_iter(&dataset, batch_size, true, None).enumerate()
            {
                let intputs = intputs.to_device(device).unwrap();
                let outputs = outputs.to_device(device).unwrap();

                let y = gpt.forward(&intputs, true);
                let y = y.reshape(vec![y.shape()[0] * y.shape()[1], y.shape()[2]]);
                // println!("y.shape = {:?}", y.shape());
                // println!("intputs.shape = {:?}", intputs.shape());
                // println!("outputs.shape = {:?}", outputs.shape());
                let outputs = outputs.reshape(vec![outputs.shape()[0] * outputs.shape()[1], 1]);
                let loss = cross_entropy(&y, &outputs).unwrap();
                println!("iter = {}, loss = {}", iter, loss.to_vec()[0]);
                let grads = loss.backward();
                optimizer.update_parameters(&mut gpt.all_trainable_parameters_map(), &grads);
            }
        }

        let gpt_safetensors = serialize(gpt.all_parameters_map()).unwrap();
        fs::write("../exclude/gpt.safetensors", gpt_safetensors).unwrap();
    }

    {
        // set_test(true);

        let gpt_safetensors = fs::read("../exclude/gpt.safetensors").unwrap();
        gpt.load_parameters_map(deserialize(gpt_safetensors, device).unwrap())
            .unwrap();

        // let input = Tensor::from_vec(vec![1, 1], vec![0.0]);
        let start = "\n";
        let start_idx = start
            .chars()
            .map(|c| dataset.tokenizer.encode(c).unwrap() as u32)
            .collect::<Vec<u32>>();
        let start_idx_len = start_idx.len();
        let input = Tensor::from_vec(start_idx, vec![1, start_idx_len], device);
        let result = generate(&mut gpt, input, 500, block_size);
        // println!("result.shape() = {:?}", result.shape());

        // let float_vec = result.to_vec();
        // let int_vec: Vec<i32> = float_vec.iter().map(|&x| x as i32).collect();
        // println!("result.to_vec() = {:?}", int_vec);

        // let int_vec: Vec<i32> = float_vec.iter().map(|&x| x as i32).collect();
        // println!("result.to_vec() = {:?}", int_vec);

        for idx in result.to_vec() {
            if let Some(chr) = dataset.tokenizer.decode(idx as usize) {
                print!("{}", chr);
            }
        }
    }
}

fn main() {
    let argv = get_argv();
    let is_gpu = if let Some(device) = argv.get("-d") {
        if device == "gpu" { true } else { false }
    } else {
        false
    };
    if is_gpu {
        #[cfg(feature = "cuda")]
        {
            run(Device::get_cuda_device())
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("cuda was not enabled.");
        }
    } else {
        run(Device::get_cpu_device())
    }
}
