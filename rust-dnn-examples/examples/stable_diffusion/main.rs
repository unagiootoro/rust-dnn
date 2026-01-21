use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten};
use rust_dnn_examples::argv::get_argv;

use crate::{clip::CLIP, diffusion::Diffusion, pipeline::{GenerateConfig, generate}, vae::VAE_Decoder};

mod vae;
mod self_attention;
mod cross_attention;
mod unet_residual_block;
mod unet_attention_block;
mod upsampling;
mod unet;
mod unet_output_layer;
mod switch_sequential;
mod time_embedding;
mod diffusion;
mod clip;
mod ddpm;
mod pipeline;

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    let cond_tokens = ten![0].to_device(device).unwrap();
    let uncond_tokens = ten![0].to_device(device).unwrap();

    println!("start create models");
    let decoder = VAE_Decoder::new(device);
    println!("decoder created");
    let clip = CLIP::new(device);
    println!("clip created");
    let diffusion = Diffusion::new(device);
    println!("diffusion created");

    println!("call generate");
    generate(GenerateConfig {
        cond_tokens,
        uncond_tokens,
        input_image: None,
        strength: 0.7,
        do_cfg: true,
        cfg_scale: 4.0,
        sampler_name: "ddpm",
        n_inference_steps: 50,
        decoder,
        clip,
        diffusion,
        seed: None,
        device,
        idle_device: Some(Device::get_cpu_device()),

    });
    Ok(())
}

fn main() -> Result<()> {
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
