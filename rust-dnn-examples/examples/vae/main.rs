use std::{
    collections::{HashMap, VecDeque},
    fs,
};

use image::{ImageBuffer, RgbImage};
use rand::prelude::*;
use rust_dnn_core::{
    backend::Backend,
    device::{self, Device},
    error::Result,
    float::Float,
    num::Num,
    ten,
    tensor::Tensor,
};
use rust_dnn_datasets::mnist::MNISTLoader;
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::batch_iter,
    layer::{BatchNorm1d, BatchNorm2d, Conv2D, Deconv2D, Layer, LayerNorm, Linear},
    loss::{cross_entropy, mean_squared_error},
    optimizer::{Adam, Optimizer, SGD},
};
use rust_dnn_safetensors::{deserialize, serialize};

pub fn flatten_except_batch<B: Backend, T: Float>(x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
    let mut dim = 1;
    for i in (1..x.ndim()).rev() {
        dim *= x.shape()[i];
    }
    x.reshape(vec![x.shape()[0], dim])
}

pub fn reshape_except_batch<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    shape: Vec<usize>,
) -> Result<Tensor<B, T>> {
    let mut shape2 = Vec::new();
    shape2.push(x.shape()[0]);
    for dim in shape {
        shape2.push(dim);
    }
    x.reshape(shape2)
}

struct Encoder<B: Backend, T: Float> {
    cv0: Conv2D<B, T>,
    cv1: Conv2D<B, T>,
    cv2: Conv2D<B, T>,
    fc0: Linear<B, T>,
    fc1_0: Linear<B, T>,
    fc1_1: Linear<B, T>,
    ln0: LayerNorm<B, T>,
    ln1: LayerNorm<B, T>,
    ln2: LayerNorm<B, T>,
    ln3: LayerNorm<B, T>,
}

impl<B: Backend, T: Float> Encoder<B, T> {
    pub fn new(z_dim: usize, device: Device<B>) -> Result<Self> {
        let cv0 = Conv2D::new(1, 32, 3, 3, 1, 1, None, true, true, device);
        let cv1 = Conv2D::new(32, 64, 2, 2, 2, 2, None, true, true, device);
        let cv2 = Conv2D::new(64, 128, 2, 2, 2, 2, None, true, true, device);
        let fc0 = Linear::new(128 * 7 * 7, 1024, true, device)?;
        let fc1_0 = Linear::new(1024, z_dim, true, device)?;
        let fc1_1 = Linear::new(1024, z_dim, true, device)?;
        let ln0 = LayerNorm::new(vec![32, 28, 28], T::from_f32(1e-4), true, device);
        let ln1 = LayerNorm::new(vec![64, 14, 14], T::from_f32(1e-4), true, device);
        let ln2 = LayerNorm::new(vec![128, 7, 7], T::from_f32(1e-4), true, device);
        let ln3 = LayerNorm::new(vec![1024], T::from_f32(1e-4), true, device);
        let encoder = Self {
            cv0,
            cv1,
            cv2,
            fc0,
            fc1_0,
            fc1_1,
            ln0,
            ln1,
            ln2,
            ln3,
        };
        Ok(encoder)
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<(Tensor<B, T>, Tensor<B, T>)> {
        let x = self.cv0.forward(x)?;
        let x = self.ln0.forward(&x)?;
        let x = x.relu();

        let x = self.cv1.forward(&x)?;
        let x = self.ln1.forward(&x)?;
        let x = x.relu();

        let x = self.cv2.forward(&x)?;
        let x = self.ln2.forward(&x)?;
        let x = x.relu();

        let x = flatten_except_batch(&x)?;
        let x = self.fc0.forward(&x)?;
        let x = self.ln3.forward(&x)?;
        let x = x.relu();

        let z_mean = self.fc1_0.forward(&x)?;
        let z_sigma = self.fc1_1.forward(&x)?;

        Ok((z_mean, z_sigma))
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Encoder<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("cv1".to_string(), &self.cv1);
        map.insert("cv2".to_string(), &self.cv2);
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1_0".to_string(), &self.fc1_0);
        map.insert("fc1_1".to_string(), &self.fc1_1);
        map.insert("ln0".to_string(), &self.ln0);
        map.insert("ln1".to_string(), &self.ln1);
        map.insert("ln2".to_string(), &self.ln2);
        map.insert("ln3".to_string(), &self.ln3);
        map
    }
}

struct Decoder<B: Backend, T: Float> {
    fc0: Linear<B, T>,
    fc1: Linear<B, T>,
    dcv0: Deconv2D<B, T>,
    dcv1: Deconv2D<B, T>,
    cv0: Conv2D<B, T>,
    ln0: LayerNorm<B, T>,
    ln1: LayerNorm<B, T>,
    ln2: LayerNorm<B, T>,
    ln3: LayerNorm<B, T>,
}

impl<B: Backend, T: Float> Decoder<B, T> {
    pub fn new(z_dim: usize, device: Device<B>) -> Result<Self> {
        let fc0 = Linear::new(z_dim, 1024, true, device)?;
        let fc1 = Linear::new(1024, 128 * 7 * 7, true, device)?;
        let dcv0 = Deconv2D::new(128, 64, 2, 2, 2, 2, None, true, true, device);
        let dcv1 = Deconv2D::new(64, 32, 2, 2, 2, 2, None, true, true, device);
        let cv0 = Conv2D::new(32, 1, 3, 3, 1, 1, None, true, true, device);
        let ln0 = LayerNorm::new(vec![1024], T::from_f32(1e-4), true, device);
        let ln1 = LayerNorm::new(vec![128 * 7 * 7], T::from_f32(1e-4), true, device);
        let ln2 = LayerNorm::new(vec![64, 14, 14], T::from_f32(1e-4), true, device);
        let ln3 = LayerNorm::new(vec![32, 28, 28], T::from_f32(1e-4), true, device);
        let decoder = Self {
            fc0,
            fc1,
            dcv0,
            dcv1,
            cv0,
            ln0,
            ln1,
            ln2,
            ln3,
        };
        Ok(decoder)
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        let x = self.fc0.forward(x)?;
        let x = self.ln0.forward(&x)?;
        let x = x.relu();

        let x = self.fc1.forward(&x)?;
        let x = self.ln1.forward(&x)?;
        let x = x.relu();

        let x = reshape_except_batch(&x, vec![128, 7, 7])?;
        let x = self.dcv0.forward(&x)?;
        let x = self.ln2.forward(&x)?;
        let x = x.relu();

        let x = self.dcv1.forward(&x)?;
        let x = self.ln3.forward(&x)?;
        let x = x.relu();

        let x = self.cv0.forward(&x)?;
        let x = x.tanh();
        Ok(x)
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Decoder<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map.insert("dcv0".to_string(), &self.dcv0);
        map.insert("dcv1".to_string(), &self.dcv1);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("ln0".to_string(), &self.ln0);
        map.insert("ln1".to_string(), &self.ln1);
        map.insert("ln2".to_string(), &self.ln2);
        map.insert("ln3".to_string(), &self.ln3);
        map
    }
}

struct VAE<B: Backend, T: Float> {
    enc: Encoder<B, T>,
    dec: Decoder<B, T>,
    z_dim: usize,
}

impl<B: Backend, T: Float> VAE<B, T> {
    pub fn new(z_dim: usize, device: Device<B>) -> Result<Self> {
        let enc = Encoder::new(z_dim, device)?;
        let dec = Decoder::new(z_dim, device)?;
        let vae = Self { enc, dec, z_dim };
        Ok(vae)
    }

    pub fn sampling(&self, z_mean: &Tensor<B, T>, z_sigma: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        let epsilon = Tensor::rand_norm(&[self.z_dim], None, z_mean.device());
        Ok((z_mean + (z_sigma * epsilon)?)?.tanh())
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<(Tensor<B, T>, Tensor<B, T>, Tensor<B, T>)> {
        let (z_mean, z_sigma) = self.enc.forward(x)?;
        let x = self.sampling(&z_mean, &z_sigma)?;
        let y = self.dec.forward(&x)?;
        Ok((y, z_mean, z_sigma))
    }
}

impl<B: Backend, T: Float> Layer<B, T> for VAE<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("enc".to_string(), &self.enc);
        map.insert("dec".to_string(), &self.dec);
        map
    }
}

fn vae_loss<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    z_mean: &Tensor<B, T>,
    z_sigma: &Tensor<B, T>,
    t: &Tensor<B, T>,
) -> Result<Tensor<B, T>> {
    // kl = -0.5 * Σ(1 + log(σ^2) - μ^2 - σ^2)
    let half = Tensor::from_scalar(T::from_f64(0.5), x.device());
    let one = Tensor::from_scalar(T::from_f64(1.0), x.device());

    let a = z_sigma.pow_scalar(T::from_f64(2.0))?.ln();
    let b = z_mean.pow_scalar(T::from_f64(2.0))?;
    let c = z_sigma.pow_scalar(T::from_f64(2.0))?;

    let d = (one + a - b - c)?;

    let kl = (-half * d.sum_axis(1, false)?.mean_axis(0, false)?)?;

    mean_squared_error(x, t)? + kl
}

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    let z_dim = 32;

    let train_dataset = MNISTLoader::new("../datasets").load_train()?;

    let model = VAE::<_, f32>::new(z_dim, device)?;
    let mut optimizer = Adam::default();

    let batch_size = 100;
    let max_epochs = 1;

    {
        for epoch in 0..max_epochs {
            println!("epoch = {}", epoch);
            for (iter, (images, _)) in
                batch_iter(&train_dataset, batch_size, true, None).enumerate()
            {
                let images = images.to_device(device)?.to_dtype::<f32>()?;
                let images =
                    ((images / ten![127.5].to_device(device)?)? - ten![1.0].to_device(device)?)?;
                let images = images.reshape(vec![100, 1, 28, 28])?;
                let (y, z_mean, z_sigma) = model.forward(&images)?;
                let loss = vae_loss(&y, &z_mean, &z_sigma, &images)?;
                let grads = loss.backward()?;
                println!("iter = {}, loss = {}", iter, loss.to_vec()[0]);
                optimizer.update_parameters(&mut model.all_trainable_parameters_map(), &grads)?;
            }

            let vae_safetensors = serialize(model.all_parameters_map())?;
            let path = format!("../exclude/vae_epoch{}.safetensors", epoch);
            fs::write(path, vae_safetensors).unwrap();
        }
    }

    {
        // let z_dim = 32;
        let mut model = VAE::<_, f32>::new(z_dim, device)?;

        let epoch = 0;
        let path = format!("../exclude/vae_epoch{}.safetensors", epoch);
        let vae_safetensors = fs::read(path).unwrap();
        model.load_parameters_map(deserialize::<B, f32>(vae_safetensors, device)?)?;

        let z = Tensor::rand_norm(&[100, z_dim], None, device);
        let y = model.dec.forward(&z)?;
        let result = ((y + 1.0)? * 127.5)?;
        let result = Tensor::cat(&[result.clone(), result.clone(), result.clone()], 1)?;
        let result = result.reshape(vec![10, 10, 3, 28, 28])?;
        let result = result
            .permuted_axes(&[0, 3, 1, 4, 2])?
            .reshape(vec![280, 280, 3])?;

        let mut pixel_data = Vec::new();
        for value in result.to_vec() {
            pixel_data.push(value as u8);
        }

        let img: RgbImage =
            ImageBuffer::from_raw(280, 280, pixel_data).expect("Invalid image buffer");
        img.save("../exclude/img.png").expect("Failed to save PNG");
    }

    {
        let test_dataset = MNISTLoader::new("../datasets").load_test()?;

        // let z_dim = 32;
        let mut model = VAE::<_, f32>::new(z_dim, device)?;

        let epoch = 0;
        let path = format!("../exclude/vae_epoch{}.safetensors", epoch);
        let vae_safetensors = fs::read(path).unwrap();
        model.load_parameters_map(deserialize::<B, f32>(vae_safetensors, device)?)?;

        let (images, _) = batch_iter(&test_dataset, batch_size, true, None)
            .next()
            .unwrap();

        let images = images.to_device(device)?.to_dtype::<f32>()?;
        let images = ((images / ten![127.5].to_device(device)?)? - ten![1.0].to_device(device)?)?;
        let images = images.reshape(vec![100, 1, 28, 28])?;
        let (y, _, _) = model.forward(&images)?;

        let result = ((y + 1.0)? * 127.5)?;
        let result = Tensor::cat(&[result.clone(), result.clone(), result.clone()], 1)?;
        let result = result.reshape(vec![10, 10, 3, 28, 28])?;
        let result = result
            .permuted_axes(&[0, 3, 1, 4, 2])?
            .reshape(vec![280, 280, 3])?;

        let mut pixel_data = Vec::new();
        for value in result.to_vec() {
            pixel_data.push(value as u8);
        }

        let img: RgbImage =
            ImageBuffer::from_raw(280, 280, pixel_data).expect("Invalid image buffer");
        img.save("../exclude/img2.png").expect("Failed to save PNG");
    }

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
