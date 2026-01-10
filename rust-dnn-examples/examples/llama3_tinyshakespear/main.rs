use std::{collections::HashMap, fs};

use rust_dnn_core::{
    backend::Backend,
    config::set_enable_backprop,
    cpu_backend::CpuBackend,
    device::{self, Device},
    float::Float,
    num::Num,
    tensor::Tensor,
};
use rust_dnn_examples::argv::get_argv;
use rust_dnn_examples::llama3::*;
use rust_dnn_nn::{
    batch_iter::{Batchable, batch_iter},
    embedding::Embedding,
    function::{generate_causal_attention_mask, rms_norm},
    layer::{Layer, LayerNorm, Linear, RMSNorm},
    layer_list::LayerList,
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

fn generate<B: Backend>(
    transformer: &mut Transformer<B>,
    mut idx: Tensor<B, u32>,
    max_new_tokens: usize,
    block_size: usize,
) -> Tensor<B, u32> {
    set_enable_backprop(false);

    for iter in 0..max_new_tokens {
        println!("generate iter = {}", iter);
        let idx_cond = if idx.size(1) <= block_size {
            idx.clone()
        } else {
            let ranges = vec![(0, idx.size(0)), (idx.size(1) - block_size, idx.size(1))];
            idx.get_item(ranges)
        };
        let logits = transformer.forward(&idx_cond, 0, false);
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

    let mut transformer = Transformer::<B>::new(
        vocab_size,
        block_size,
        n_layers,
        n_embd,
        n_head,
        n_embd * 4,
        true,
        false,
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

                let y = transformer.forward(&intputs, 0, true);
                let y = y.reshape(vec![y.shape()[0] * y.shape()[1], y.shape()[2]]);
                // println!("y.shape = {:?}", y.shape());
                // println!("intputs.shape = {:?}", intputs.shape());
                // println!("outputs.shape = {:?}", outputs.shape());
                let outputs = outputs.reshape(vec![outputs.shape()[0] * outputs.shape()[1]]);
                let loss = cross_entropy(&y, &outputs).unwrap();
                println!("iter = {}, loss = {}", iter, loss.to_vec()[0]);
                let grads = loss.backward();
                optimizer
                    .update_parameters(&mut transformer.all_trainable_parameters_map(), &grads);
            }
        }

        let gpt_safetensors = serialize(transformer.all_parameters_map()).unwrap();
        fs::write("../exclude/llama3.safetensors", gpt_safetensors).unwrap();
    }

    {
        // set_test(true);

        let gpt_safetensors = fs::read("../exclude/llama3.safetensors").unwrap();
        transformer
            .load_parameters_map(deserialize(gpt_safetensors, device).unwrap())
            .unwrap();

        // let input = Tensor::from_vec(vec![1, 1], vec![0.0]);
        let start = "\n";
        let start_idx = start
            .chars()
            .map(|c| dataset.tokenizer.encode(c).unwrap() as u32)
            .collect::<Vec<u32>>();
        let start_idx_len = start_idx.len();
        let input = Tensor::from_vec(start_idx, vec![1, start_idx_len], device);
        let result = generate(&mut transformer, input, 500, block_size);
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
