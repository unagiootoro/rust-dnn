use core::time;

use rust_dnn_core::{backend::Backend, device::Device, tensor::Tensor};

pub struct DDPMSampler<B: Backend> {
    betas: Tensor<B, f32>,
    alphas: Tensor<B, f32>,
    alphas_cumprod: Tensor<B, f32>,
    one: Tensor<B, f32>,
    timesteps: Tensor<B, f32>,
    num_train_timesteps: usize,
    num_inference_steps: usize,
    start_step: usize,
    device: Device<B>,
}

impl<B: Backend> DDPMSampler<B> {
    // def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
    pub fn new(device: Device<B>) -> Self {
        Self::new2(1000, 0.00085, 0.0120, device)
    }

    pub fn new2(
        num_training_steps: usize,
        beta_start: f64,
        beta_end: f64,
        device: Device<B>,
    ) -> Self {
        //     self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        let betas = Tensor::linspace(
            beta_start.powf(0.5) as f32,
            beta_end.powf(0.5) as f32,
            num_training_steps,
            device,
        ).pow_scalar(2.0);
        //     self.alphas = 1.0 - self.betas
        let alphas = 1.0 - &betas;
        //     self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        let alphas_cumprod = alphas.cumprod(0);
        //     self.one = torch.tensor(1.0)
        let one = Tensor::from_f64(1.0, device);
        //     self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())
        let timesteps = Tensor::arange(0..(num_training_steps as isize), device);
        Self {
            betas,
            alphas,
            alphas_cumprod,
            one,
            timesteps,
            num_train_timesteps: num_training_steps,
            num_inference_steps: 1,
            start_step: 0,
            device,
        }
    }

    pub fn timesteps(&self) -> &Tensor<B, f32> {
        &self.timesteps
    }

    pub fn set_inference_timesteps(&mut self, num_inference_steps: usize) {
        self.num_inference_steps = num_inference_steps;
        let step_ratio = self.num_train_timesteps / num_inference_steps;
        let timesteps = Tensor::<B, f32>::arange(0..(num_inference_steps as isize), self.device);
        let timesteps = (timesteps * step_ratio as f64).round().flip(&[0]);
        self.timesteps = timesteps;
    }

    fn get_previous_timestep(&self, timestep: usize) -> isize {
        timestep as isize - self.num_train_timesteps as isize / self.num_inference_steps as isize
    }

    fn get_variance(&self, timestep: usize) -> Tensor<B, f32> {
        let prev_t = self.get_previous_timestep(timestep);
        let alpha_prod_t = self.alphas_cumprod.select2(0, timestep as isize);
        let alpha_prod_t_prev = self.alphas_cumprod.select2(0, prev_t as isize);
        let current_beta_t = 1.0 - &alpha_prod_t / &alpha_prod_t_prev;
        let variance = (1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * current_beta_t;
        variance.maximum_scalar(1e-20)
    }

    pub fn set_strength(&mut self, strength: usize) {
        let start_step = self.num_inference_steps - self.num_inference_steps * strength;
        self.timesteps = self
            .timesteps
            .narrow(0, start_step, self.timesteps.size(0) - start_step);
        self.start_step = start_step
    }

    pub fn step(
        &mut self,
        timestep: usize,
        latents: Tensor<B, f32>,
        model_output: Tensor<B, f32>,
    ) -> Tensor<B, f32> {
        //     t = timestep
        let t = timestep;
        //     prev_t = self._get_previous_timestep(t)
        let prev_t = self.get_previous_timestep(t);

        //     # 1. compute alphas, betas
        //     alpha_prod_t = self.alphas_cumprod[t]
        let alpha_prod_t = self.alphas_cumprod.select2(0, t as isize);
        //     alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        let alpha_prod_t_prev = if prev_t >= 0 {
            self.alphas_cumprod.select2(0, prev_t as isize)
        } else {
            self.one.clone()
        };
        // println!("timestep = {:?}", timestep);
        // println!("prev_t = {:?}", prev_t);
        // println!("alpha_prod_t = {:?}", &alpha_prod_t.to_vec());
        // println!("alpha_prod_t_prev = {:?}", &alpha_prod_t_prev.to_vec());
        //     beta_prod_t = 1 - alpha_prod_t
        let beta_prod_t = 1.0 - &alpha_prod_t;
        //     beta_prod_t_prev = 1 - alpha_prod_t_prev
        let beta_prod_t_prev = 1.0 - &alpha_prod_t_prev;
        //     current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        let current_alpha_t = &alpha_prod_t / &alpha_prod_t_prev;
        //     current_beta_t = 1 - current_alpha_t
        let current_beta_t = 1.0 - &current_alpha_t;

        //     # 2. compute predicted original sample from predicted noise also called
        //     # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        //     pred_original_sample = (latents - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        let pred_original_sample =
            (&latents - beta_prod_t.pow_scalar(0.5) * &model_output) / alpha_prod_t.pow_scalar(0.5);

        //     # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        //     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        //     pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        let pred_original_sample_coeff =
            (&alpha_prod_t_prev.pow_scalar(0.5) * current_beta_t) / &beta_prod_t;
        //     current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        let current_sample_coeff =
            (&current_alpha_t.pow_scalar(0.5) * beta_prod_t_prev) / beta_prod_t;

        //     # 5. Compute predicted previous sample µ_t
        //     # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        //     pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents
        let pred_prev_sample =
            pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents;

        //     # 6. Add noise
        //     variance = 0
        //     if t > 0:
        let variance = if t > 0 {
            //         device = model_output.device
            //         noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            let noise = Tensor::rand_norm(model_output.shape(), None, model_output.device());
            //         # Compute the variance as per formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            //         variance = (self._get_variance(t) ** 0.5) * noise
            (self.get_variance(t).pow_scalar(0.5)) * noise
            // (self.get_variance(t).pow_scalar(0.5))
        } else {
            Tensor::zeros(vec![1], model_output.device())
        };

        //     # sample from N(mu, sigma) = X can be obtained by X = mu + sigma * N(0, 1)
        //     # the variable "variance" is already multiplied by the noise N(0, 1)
        //     pred_prev_sample = pred_prev_sample + variance
        println!("pred_prev_sample = {:?}", &pred_prev_sample.to_vec()[0..8]);
        println!("variance = {:?}", &variance.to_vec());
        let pred_prev_sample = pred_prev_sample + variance;

        //     return pred_prev_sample
        pred_prev_sample
    }

    pub fn add_noise(
        &self,
        original_samples: Tensor<B, f32>,
        timesteps: Tensor<B, u32>,
    ) -> Tensor<B, f32> {
        let sqrt_alpha_prod = self.alphas_cumprod.index_select(0, &timesteps);
        let mut sqrt_alpha_prod = sqrt_alpha_prod.flatten_all();
        while sqrt_alpha_prod.ndim() < original_samples.ndim() {
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1);
        }

        let sqrt_one_minus_alpha_prod =
            (1.0 - self.alphas_cumprod.index_select(0, &timesteps)).pow_scalar(0.5);
        let mut sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten_all();
        while sqrt_one_minus_alpha_prod.ndim() < original_samples.ndim() {
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1);
        }

        let noise = Tensor::rand_norm(original_samples.shape(), None, original_samples.device());
        let noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise;
        noisy_samples
    }
}
