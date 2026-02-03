use std::fs;

use image::{ImageBuffer, RgbImage};
use rust_dnn_core::{backend::Backend, device::Device, error::Result, ten};
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::layer::Layer;
use rust_dnn_safetensors::deserialize;

use crate::{
    clip::CLIP,
    diffusion::Diffusion,
    pipeline::{GenerateConfig, generate},
    vae::VAE_Decoder,
};

mod clip;
mod cross_attention;
mod ddpm;
mod diffusion;
mod pipeline;
mod self_attention;
mod switch_sequential;
mod time_embedding;
mod unet;
mod unet_attention_block;
mod unet_output_layer;
mod unet_residual_block;
mod upsampling;
mod vae;

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    let cond_tokens = ten![
        49406, 1579, 33313, 267, 25602, 267, 272, 271, 271, 2848, 8666, 267, 279, 330, 9977, 269,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407
    ]
    .to_device(device)
    .unwrap();

    let uncond_tokens = ten![
        49406, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
        49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407
    ]
    .to_device(device)
    .unwrap();

    println!("start create models");
    let mut decoder = VAE_Decoder::new(device);
    println!("decoder created");
    let mut clip = CLIP::new(device);
    println!("clip created");
    let mut diffusion = Diffusion::new(device);
    println!("diffusion created");

    {
        let sd1_5_decoder_safetensors = fs::read("../exclude/sd1_5_decoder.safetensors").unwrap();
        println!("end load sd1_5_decoder.safetensors");
        decoder
            .load_parameters_map(deserialize(sd1_5_decoder_safetensors, device).unwrap())
            .unwrap();
        println!("end load sd1_5_decoder weights");
    }

    {
        let sd1_5_clip_safetensors = fs::read("../exclude/sd1_5_clip.safetensors").unwrap();
        println!("end load sd1_5_clip.safetensors");
        clip.load_parameters_map(deserialize(sd1_5_clip_safetensors, device).unwrap())
            .unwrap();
        println!("end load sd1_5_clip weights");
    }

    {
        let sd1_5_diffusion_safetensors =
            fs::read("../exclude/sd1_5_diffusion.safetensors").unwrap();
        println!("end load sd1_5_diffusion.safetensors");
        diffusion
            .load_parameters_map(deserialize(sd1_5_diffusion_safetensors, device).unwrap())
            .unwrap();
        println!("end load sd1_5_diffusion weights");
    }

    println!("call generate");
    let output_image = generate(GenerateConfig {
        cond_tokens,
        uncond_tokens,
        input_image: None,
        strength: 0.9,
        do_cfg: true,
        cfg_scale: 8.0,
        sampler_name: "ddpm",
        n_inference_steps: 50,
        decoder,
        clip,
        diffusion,
        seed: None,
        device,
        idle_device: Some(Device::get_cpu_device()),
    });

    let mut pixel_data = Vec::new();
    for value in output_image.to_vec() {
        let value = value.clamp(0.0, 255.0);
        pixel_data.push(value as u8);
    }

    let img: RgbImage = ImageBuffer::from_raw(512, 512, pixel_data).expect("Invalid image buffer");
    img.save("../exclude/output_image.png")
        .expect("Failed to save PNG");

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
