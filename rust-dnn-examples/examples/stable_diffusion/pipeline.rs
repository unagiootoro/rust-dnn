use rust_dnn_core::{
    backend::Backend,
    config::{enable_backprop, set_enable_backprop},
    cpu_backend::CpuBackend,
    device::Device,
    num::Num,
    ten,
    tensor::Tensor,
};

use crate::{
    clip::{self, CLIP},
    ddpm::DDPMSampler,
    debug_latent_data::DEBUG_LATENT_INPUT,
    diffusion::Diffusion,
    vae::VAE_Decoder,
};

const WIDTH: usize = 512;
const HEIGHT: usize = 512;
const LATENTS_WIDTH: usize = WIDTH / 8;
const LATENTS_HEIGHT: usize = HEIGHT / 8;

// def generate(
//     prompt,
//     uncond_prompt=None,
//     input_image=None,
//     strength=0.8,
//     do_cfg=True,
//     cfg_scale=7.5,
//     sampler_name="ddpm",
//     n_inference_steps=50,
//     models={},
//     seed=None,
//     device=None,
//     idle_device=None,
//     tokenizer=None,
// ):

pub struct GenerateConfig<'a, B: Backend> {
    pub cond_tokens: Tensor<B, u32>,
    pub uncond_tokens: Tensor<B, u32>,
    pub input_image: Option<()>, // TODO:
    pub strength: f64,
    pub do_cfg: bool,
    pub cfg_scale: f64,
    pub sampler_name: &'a str,
    pub n_inference_steps: usize,
    pub clip: CLIP<B>,
    pub diffusion: Diffusion<B>,
    pub decoder: VAE_Decoder<B>,
    pub seed: Option<u64>,
    pub device: Device<B>,
    pub idle_device: Option<Device<CpuBackend>>,
}

pub fn generate<'a, B: Backend>(mut cfg: GenerateConfig<B>) -> Tensor<CpuBackend, f32> {
    //     with torch.no_grad():
    let prev_enable_backprop = enable_backprop();
    set_enable_backprop(false);

    //         if not 0 < strength <= 1:
    //             raise ValueError("strength must be between 0 and 1")
    if !(0.0 < cfg.strength && cfg.strength <= 1.0) {
        panic!("strength must be between 0 and 1");
    }

    //         if idle_device:
    //             to_idle = lambda x: x.to(idle_device)
    //         else:
    //             to_idle = lambda x: x

    // TODO:
    // if let Some(idle_device) = cfg.idle_device {

    // } else {

    // };

    //         # Initialize random number generator according to the seed specified
    //         generator = torch.Generator(device=device)
    //         if seed is None:
    //             generator.seed()
    //         else:
    //             generator.manual_seed(seed)

    //         clip = models["clip"]
    //         clip.to(device)

    let context = if cfg.do_cfg {
        //             # Convert into a list of length Seq_Len=77
        //             cond_tokens = tokenizer.batch_encode_plus(
        //                 [prompt], padding="max_length", max_length=77
        //             ).input_ids
        //             # (Batch_Size, Seq_Len)
        //             cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        //             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        //             cond_context = clip(cond_tokens)
        let cond_tokens = cfg.cond_tokens.reshape(&vec![1isize, -1isize]);
        let cond_context = cfg.clip.forward(&cond_tokens);

        //             # Convert into a list of length Seq_Len=77
        //             uncond_tokens = tokenizer.batch_encode_plus(
        //                 [uncond_prompt], padding="max_length", max_length=77
        //             ).input_ids
        //             # (Batch_Size, Seq_Len)
        //             uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        //             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        //             uncond_context = clip(uncond_tokens)
        let uncond_tokens = cfg.uncond_tokens.reshape(&vec![1isize, -1isize]);
        let uncond_context = cfg.clip.forward(&uncond_tokens);

        //             # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
        //             context = torch.cat([cond_context, uncond_context])
        Tensor::cat(&[cond_context, uncond_context], 0)
    } else {
        //             # Convert into a list of length Seq_Len=77
        //             tokens = tokenizer.batch_encode_plus(
        //                 [prompt], padding="max_length", max_length=77
        //             ).input_ids
        //             # (Batch_Size, Seq_Len)
        //             tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        //             # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        //             context = clip(tokens)
        cfg.clip.forward(&cfg.cond_tokens)
    };
    //         to_idle(clip)

    //         if sampler_name == "ddpm":
    //             sampler = DDPMSampler(generator)
    //             sampler.set_inference_timesteps(n_inference_steps)
    //         else:
    //             raise ValueError("Unknown sampler value %s. ")
    if cfg.sampler_name != "ddpm" {
        panic!("Unknown sampler value {}. ", cfg.sampler_name);
    }

    let mut sampler = DDPMSampler::new(cfg.device);
    sampler.set_inference_timesteps(cfg.n_inference_steps);

    //         latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    let latents_shape = vec![1, 4, LATENTS_HEIGHT, LATENTS_WIDTH];

    //         if input_image:
    //             encoder = models["encoder"]
    //             encoder.to(device)

    //             input_image_tensor = input_image.resize((WIDTH, HEIGHT))
    //             # (Height, Width, Channel)
    //             input_image_tensor = np.array(input_image_tensor)
    //             # (Height, Width, Channel) -> (Height, Width, Channel)
    //             input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
    //             # (Height, Width, Channel) -> (Height, Width, Channel)
    //             input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
    //             # (Height, Width, Channel) -> (Batch_Size, Height, Width, Channel)
    //             input_image_tensor = input_image_tensor.unsqueeze(0)
    //             # (Batch_Size, Height, Width, Channel) -> (Batch_Size, Channel, Height, Width)
    //             input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

    //             # (Batch_Size, 4, Latents_Height, Latents_Width)
    //             encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
    //             # (Batch_Size, 4, Latents_Height, Latents_Width)
    //             latents = encoder(input_image_tensor, encoder_noise)

    //             # Add noise to the latents (the encoded input image)
    //             # (Batch_Size, 4, Latents_Height, Latents_Width)
    //             sampler.set_strength(strength=strength)
    //             latents = sampler.add_noise(latents, sampler.timesteps[0])

    //             to_idle(encoder)
    //         else:
    //             # (Batch_Size, 4, Latents_Height, Latents_Width)
    //             latents = torch.randn(latents_shape, generator=generator, device=device)
    // let mut latents = Tensor::rand_norm(&latents_shape, cfg.seed, cfg.device);
    let mut latents = Tensor::from_vec(DEBUG_LATENT_INPUT.to_vec(), latents_shape, cfg.device);

    //         diffusion = models["diffusion"]
    //         diffusion.to(device)

    //         timesteps = tqdm(sampler.timesteps)
    let timesteps = sampler.timesteps();

    //         for i, timestep in enumerate(timesteps):
    for (i, timestep) in timesteps.to_vec().iter().enumerate() {
        println!("iter = {}", i);
        //             # (1, 320)
        //             time_embedding = get_time_embedding(timestep).to(device)
        let time_embedding = get_time_embedding(*timestep as usize, cfg.device);

        //             # (Batch_Size, 4, Latents_Height, Latents_Width)
        //             model_input = latents
        let mut model_input = latents.clone();

        //             if do_cfg:
        //                 # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
        //                 model_input = model_input.repeat(2, 1, 1, 1)
        if cfg.do_cfg {
            model_input = model_input.repeat(&[2, 1, 1, 1]);
        }

        //             # model_output is the predicted noise
        //             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
        //             model_output = diffusion(model_input, context, time_embedding)
        let mut model_output = cfg
            .diffusion
            .forward(&model_input, &context, &time_embedding);
        println!("model_output = {:?}", &model_output.to_vec()[0..8]);

        //             if do_cfg:
        //                 output_cond, output_uncond = model_output.chunk(2)
        //                 model_output = cfg_scale * (output_cond - output_uncond) + output_uncond
        if cfg.do_cfg {
            let chunks = model_output.chunk(0, 2);
            let output_cond = &chunks[0];
            let output_uncond = &chunks[1];
            model_output = cfg.cfg_scale * (output_cond - output_uncond) + output_uncond;
            println!("2: model_output = {:?}", &model_output.to_vec()[0..8]);
        }

        //             # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
        //             latents = sampler.step(timestep, latents, model_output)
        latents = sampler.step(*timestep as usize, latents, model_output);
        println!("latents = {:?}", &latents.to_vec()[0..8]);
    }

    //         to_idle(diffusion)

    //         decoder = models["decoder"]
    //         decoder.to(device)
    //         # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
    //         images = decoder(latents)
    let images = cfg.decoder.forward(&latents);
    //         to_idle(decoder)

    //         images = rescale(images, (-1, 1), (0, 255), clamp=True)
    let images = rescale(&images, (-1.0, 1.0), (0.0, 255.0), true);
    //         # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
    //         images = images.permute(0, 2, 3, 1)
    let images = images.permuted_axes(&[0, 2, 3, 1]);
    //         images = images.to("cpu", torch.uint8).numpy()
    //         return images[0]
    let images = images.to_device(Device::get_cpu_device()).unwrap();

    set_enable_backprop(prev_enable_backprop);
    images
}

fn rescale<B: Backend>(
    x: &Tensor<B, f32>,
    old_range: (f64, f64),
    new_range: (f64, f64),
    is_clamp: bool,
) -> Tensor<B, f32> {
    let mut x = x.clone();
    let (old_min, old_max) = old_range;
    let (new_min, new_max) = new_range;
    x = x - old_min;
    x = x * ((new_max - new_min) / (old_max - old_min));
    x = x + new_min;
    if is_clamp {
        x = x.clamp_scalar(new_min, new_max);
    }
    x
}

fn get_time_embedding<B: Backend>(timestep: usize, device: Device<B>) -> Tensor<B, f32> {
    //     # Shape: (160,)
    //     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    let v = ten![10000.0].to_device(device).unwrap();
    let freqs = v.pow(&(-Tensor::arange(0..160, device) / 160.0));

    //     # Shape: (1, 160)
    //     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    let x = ten![timestep as f32]
        .to_device(device)
        .unwrap()
        .unsqueeze(-1)
        * freqs.unsqueeze(0);

    //     # Shape: (1, 160 * 2)
    //     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
    Tensor::cat(&[x.cos(), x.sin()], -1)
}
