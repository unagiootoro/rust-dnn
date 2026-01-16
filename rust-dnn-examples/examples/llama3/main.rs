use core::f32;
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
use rust_dnn_examples::{argv::get_argv, llama3::Transformer};
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

fn run<B: Backend>(device: Device<B>) {
    let vocab_size = 128256;
    let block_size = 2048;
    let n_layers = 16;
    let n_embd = 2048;
    let n_head = 32;

    let mut transformer = Transformer::<B>::new(
        vocab_size,
        block_size,
        n_layers,
        n_embd,
        n_head,
        8192,
        true,
        true,
        device,
    );

    {
        set_enable_backprop(false);

        {
            println!("start load llama3 safetensors");
            let llama3_safetensors = fs::read("../exclude/llama3_1b.safetensors").unwrap();
            println!("end load llama3 safetensors");

            println!("start load llama3 weights");
            transformer
                .load_parameters_map(deserialize(llama3_safetensors, device).unwrap())
                .unwrap();
            println!("end load llama3 weights");
        }

        {
            let start_idx = vec![128000, 791, 6864, 315, 6457, 374];
            let start_idx_len = start_idx.len();
            let mut input_ids = Tensor::from_vec(start_idx, vec![1, start_idx_len], device);

            let mut curr_pos = 0;
            for i in 0..30 {
                println!("generate iter = {}", i);
                let ranges = vec![(0, input_ids.size(0)), (curr_pos, input_ids.size(1))];
                let idx = input_ids.get_item(ranges);
                let logits: Tensor<B, f32> = transformer.forward(&idx, curr_pos, false);
                let next_token = logits.select(1, logits.size(1) - 1).argmax_axis(-1, false);
                input_ids = Tensor::cat(&vec![input_ids, next_token.unsqueeze(0)], 1);
                curr_pos = input_ids.size(1) - 1;
            }

            println!("result = {:?}", input_ids.to_vec());
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
