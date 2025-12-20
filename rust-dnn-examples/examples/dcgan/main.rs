use std::{collections::HashMap, fs};

use image::{ImageBuffer, RgbImage};
use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, ten, tensor::Tensor,
};
use rust_dnn_datasets::mnist::MNISTLoader;
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::batch_iter,
    layer::{BatchNorm1d, BatchNorm2d, Conv2D, Deconv2D, Layer, Linear},
    loss::sigmoid_cross_entropy,
    optimizer::{Adam, Optimizer},
};
use rust_dnn_safetensors::{deserialize, serialize};

pub fn flatten_except_batch<B: Backend, T: Float>(x: &Tensor<B, T>) -> Tensor<B, T> {
    let mut dim = 1;
    for i in (1..x.ndim()).rev() {
        dim *= x.shape()[i];
    }
    x.reshape(vec![x.shape()[0], dim])
}

pub fn reshape_except_batch<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    shape: Vec<usize>,
) -> Tensor<B, T> {
    let mut shape2 = Vec::new();
    shape2.push(x.shape()[0]);
    for dim in shape {
        shape2.push(dim);
    }
    x.reshape(shape2)
}

struct Generator<B: Backend> {
    fc0: Linear<B, f32>,
    fc1: Linear<B, f32>,
    dcv0: Deconv2D<B, f32>,
    dcv1: Deconv2D<B, f32>,
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm1d<B, f32>,
    bn1: BatchNorm1d<B, f32>,
    bn2: BatchNorm2d<B, f32>,
    bn3: BatchNorm2d<B, f32>,
}

impl<B: Backend> Generator<B> {
    pub fn new(z_dim: usize, device: Device<B>) -> Self {
        let fc0 = Linear::new(z_dim, 1024, true, device);
        let fc1 = Linear::new(1024, 128 * 7 * 7, true, device);
        let dcv0 = Deconv2D::new(128, 64, 2, 2, 2, 2, None, true, true, device);
        let dcv1 = Deconv2D::new(64, 32, 2, 2, 2, 2, None, true, true, device);
        let cv0 = Conv2D::new(32, 1, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm1d::new(1024, 0.9, 1e-4, device);
        let bn1 = BatchNorm1d::new(128 * 7 * 7, 0.9, 1e-4, device);
        let bn2 = BatchNorm2d::new(64, 0.9, 1e-4, device);
        let bn3 = BatchNorm2d::new(32, 0.9, 1e-4, device);
        Self {
            fc0,
            fc1,
            dcv0,
            dcv1,
            cv0,
            bn0,
            bn1,
            bn2,
            bn3,
        }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Tensor<B, f32> {
        let x = self.fc0.forward(x);
        let x = self.bn0.forward(&x, is_train);
        let x = x.relu();

        let x = self.fc1.forward(&x);
        let x = self.bn1.forward(&x, is_train);
        let x = x.relu();

        let x = reshape_except_batch(&x, vec![128, 7, 7]);
        let x = self.dcv0.forward(&x);
        let x = self.bn2.forward(&x, is_train);
        let x = x.relu();

        let x = self.dcv1.forward(&x);
        let x = self.bn3.forward(&x, is_train);
        let x = x.relu();

        let x = self.cv0.forward(&x);
        let x = x.tanh();
        x
    }
}

impl<B: Backend> Layer<B, f32> for Generator<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map.insert("dcv0".to_string(), &self.dcv0);
        map.insert("dcv1".to_string(), &self.dcv1);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map.insert("bn1".to_string(), &self.bn1);
        map.insert("bn2".to_string(), &self.bn2);
        map.insert("bn3".to_string(), &self.bn3);
        map
    }
}

struct Discriminator<B: Backend> {
    cv0: Conv2D<B, f32>,
    cv1: Conv2D<B, f32>,
    cv2: Conv2D<B, f32>,
    fc0: Linear<B, f32>,
    fc1: Linear<B, f32>,
}

impl<B: Backend> Discriminator<B> {
    pub fn new(z_dim: usize, device: Device<B>) -> Self {
        let cv0 = Conv2D::new(1, 32, 3, 3, 1, 1, None, true, true, device);
        let cv1 = Conv2D::new(32, 64, 2, 2, 2, 2, None, true, true, device);
        let cv2 = Conv2D::new(64, 128, 2, 2, 2, 2, None, true, true, device);
        let fc0 = Linear::new(128 * 7 * 7, 1024, true, device);
        let fc1 = Linear::new(1024, 1, true, device);
        Self {
            cv0,
            cv1,
            cv2,
            fc0,
            fc1,
        }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(x);
        let x = x.leaky_relu(0.2);

        let x = self.cv1.forward(&x);
        let x = x.leaky_relu(0.2);

        let x = self.cv2.forward(&x);
        let x = x.leaky_relu(0.2);

        let x = flatten_except_batch(&x);
        let x = self.fc0.forward(&x);
        let x = x.leaky_relu(0.2);

        let x = self.fc1.forward(&x);
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for Discriminator<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("cv1".to_string(), &self.cv1);
        map.insert("cv2".to_string(), &self.cv2);
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map
    }
}

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    let z_dim = 32;

    let train_dataset = MNISTLoader::new("../datasets").load_train()?;

    let mut generator = Generator::new(z_dim, device);
    let mut discriminator = Discriminator::new(z_dim, device);
    let mut opt_gen = Adam::new(0.0002, 0.5, 0.999, 1e-7);
    let mut opt_dis = Adam::new(0.0002, 0.5, 0.999, 1e-7);

    let batch_size = 100;
    let max_epochs = 1;

    let label_real = Tensor::ones(vec![batch_size, 1], device);
    let label_fake = Tensor::zeros(vec![batch_size, 1], device);

    {
        for epoch in 0..max_epochs {
            println!("epoch = {}", epoch);
            for (iter, (images, _)) in
                batch_iter(&train_dataset, batch_size, true, None).enumerate()
            {
                let images = images.to_device(device)?.to_dtype::<f32>()?;
                let images =
                    ((images / ten![127.5].to_device(device)?) - ten![1.0].to_device(device)?);
                let images = images.reshape(vec![100, 1, 28, 28]);

                let z = Tensor::rand_norm(&[batch_size, z_dim], None, device);
                let fake = generator.forward(&z, true);
                let y_real = discriminator.forward(&images)?;
                let y_fake = discriminator.forward(&fake.detach())?;
                let loss_dis_real = sigmoid_cross_entropy(&y_real, &label_real)?;
                let loss_dis_fake = sigmoid_cross_entropy(&y_fake, &label_fake)?;
                let loss_dis = loss_dis_real + loss_dis_fake;
                let grads = loss_dis.backward();
                opt_dis
                    .update_parameters(&mut discriminator.all_trainable_parameters_map(), &grads);

                let y_fake = discriminator.forward(&fake)?;
                let loss_gen = sigmoid_cross_entropy(&y_fake, &label_real)?;
                let grads = loss_gen.backward();
                opt_gen.update_parameters(&mut generator.all_trainable_parameters_map(), &grads);

                println!(
                    "iter = {}, loss_dis = {}, loss_gen = {}",
                    iter,
                    loss_dis.to_vec()[0],
                    loss_gen.to_vec()[0]
                );
            }

            let vae_safetensors = serialize(generator.all_parameters_map())?;
            let path = format!("../exclude/dcgan_epoch{}.safetensors", epoch);
            fs::write(path, vae_safetensors).unwrap();
        }
    }

    {
        let mut generator = Generator::new(z_dim, device);

        let epoch = 0;
        let path = format!("../exclude/dcgan_epoch{}.safetensors", epoch);
        let vae_safetensors = fs::read(path).unwrap();
        generator.load_parameters_map(deserialize::<B, f32>(vae_safetensors, device)?)?;

        let z = Tensor::rand_norm(&[100, z_dim], None, device);
        let y = generator.forward(&z, false);
        let result = (y + 1.0) * 127.5;
        let result = Tensor::cat(&[result.clone(), result.clone(), result.clone()], 1);
        let result = result.reshape(vec![10, 10, 3, 28, 28]);
        let result = result
            .permuted_axes(&[0, 3, 1, 4, 2])
            .reshape(vec![280, 280, 3]);

        let mut pixel_data = Vec::new();
        for value in result.to_vec() {
            pixel_data.push(value as u8);
        }

        let img: RgbImage =
            ImageBuffer::from_raw(280, 280, pixel_data).expect("Invalid image buffer");
        img.save("../exclude/img.png").expect("Failed to save PNG");
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
