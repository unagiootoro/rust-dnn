use std::{collections::HashMap, f32, fs};

use image::{ImageBuffer, RgbImage};
use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, ten, tensor::Tensor,
};
use rust_dnn_datasets::cifar10::CIFAR10Loader;
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::batch_iter,
    layer::{BatchNorm1d, BatchNorm2d, Conv2D, Deconv2D, Layer, Linear},
    loss::sigmoid_cross_entropy,
    optimizer::{Adam, Optimizer},
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

struct UpConvBlock<B: Backend> {
    dcv0: Deconv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> UpConvBlock<B> {
    pub fn new(in_features: usize, out_features: usize, device: Device<B>) -> Self {
        let dcv0 = Deconv2D::new(
            in_features,
            out_features,
            2,
            2,
            2,
            2,
            None,
            true,
            true,
            device,
        );
        let bn0 = BatchNorm2d::new(out_features, 0.9, 1e-4, device);
        Self { dcv0, bn0 }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.dcv0.forward(x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.relu();
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for UpConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("dcv0".to_string(), &self.dcv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct DownConvBlock<B: Backend> {
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> DownConvBlock<B> {
    pub fn new(in_features: usize, out_features: usize, device: Device<B>) -> Self {
        let cv0 = Conv2D::new(
            in_features,
            out_features,
            2,
            2,
            2,
            2,
            None,
            true,
            true,
            device,
        );
        let bn0 = BatchNorm2d::new(out_features, 0.9, 1e-4, device);
        Self { cv0, bn0 }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(&x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.relu();
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for DownConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct ConvBlock<B: Backend> {
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(features: usize, device: Device<B>) -> Self {
        let cv0 = Conv2D::new(features, features, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        Self { cv0, bn0 }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(&x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.relu();
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for ConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct Generator<B: Backend> {
    down_conv_block0: DownConvBlock<B>,
    down_conv_block1: DownConvBlock<B>,
    conv_block0: ConvBlock<B>,
    up_conv_block0: UpConvBlock<B>,
    up_conv_block1: UpConvBlock<B>,
    cv0: Conv2D<B, f32>,
    cv1: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> Generator<B> {
    pub fn new(features: usize, device: Device<B>) -> Result<Self> {
        let down_conv_block0 = DownConvBlock::new(features, features * 2, device);
        let down_conv_block1 = DownConvBlock::new(features * 2, features * 4, device);
        let conv_block0 = ConvBlock::new(features * 4, device);
        let up_conv_block0 = UpConvBlock::new(features * 4 + features * 4, features * 2, device);
        let up_conv_block1 = UpConvBlock::new(features * 2 + features * 2, features, device);
        let cv0 = Conv2D::new(1, features, 3, 3, 1, 1, None, true, true, device);
        let cv1 = Conv2D::new(features, 3, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        let generator = Self {
            down_conv_block0,
            down_conv_block1,
            conv_block0,
            up_conv_block0,
            up_conv_block1,
            cv0,
            cv1,
            bn0,
        };
        Ok(generator)
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.relu();

        let x = self.down_conv_block0.forward(&x, is_train)?;
        let x0 = &x;
        let x = self.down_conv_block1.forward(&x, is_train)?;
        let x1 = &x;
        let x = self.conv_block0.forward(&x, is_train)?;
        let x = Tensor::cat(&[x1.clone(), x.clone()], 1)?;
        let x = self.up_conv_block0.forward(&x, is_train)?;
        let x = Tensor::cat(&[x0.clone(), x.clone()], 1)?;
        let x = self.up_conv_block1.forward(&x, is_train)?;

        let x = self.cv1.forward(&x)?;
        let x = x.tanh();
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for Generator<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("down_conv_block0".to_string(), &self.down_conv_block0);
        map.insert("down_conv_block1".to_string(), &self.down_conv_block1);
        map.insert("conv_block0".to_string(), &self.conv_block0);
        map.insert("up_conv_block0".to_string(), &self.up_conv_block0);
        map.insert("up_conv_block1".to_string(), &self.up_conv_block1);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("cv1".to_string(), &self.cv1);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct DiscriminatorDownConvBlock<B: Backend> {
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> DiscriminatorDownConvBlock<B> {
    pub fn new(in_features: usize, out_features: usize, device: Device<B>) -> Self {
        let cv0 = Conv2D::new(
            in_features,
            out_features,
            2,
            2,
            2,
            2,
            None,
            true,
            true,
            device,
        );
        let bn0 = BatchNorm2d::new(out_features, 0.9, 1e-4, device);
        Self { cv0, bn0 }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.leaky_relu(0.2);
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for DiscriminatorDownConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct DiscriminatorConvBlock<B: Backend> {
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> DiscriminatorConvBlock<B> {
    pub fn new(features: usize, device: Device<B>) -> Self {
        let cv0 = Conv2D::new(features, features, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        Self { cv0, bn0 }
    }

    pub fn forward(&mut self, x: &Tensor<B, f32>, is_train: bool) -> Result<Tensor<B, f32>> {
        let x = self.cv0.forward(x)?;
        let x = self.bn0.forward(&x, is_train)?;
        let x = x.leaky_relu(0.2);
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for DiscriminatorConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct Discriminator<B: Backend> {
    cv0: Conv2D<B, f32>,
    cv1: Conv2D<B, f32>,
    cv2: Conv2D<B, f32>,
    down_conv_block0: DiscriminatorDownConvBlock<B>,
    down_conv_block1: DiscriminatorDownConvBlock<B>,
    conv_block0: DiscriminatorConvBlock<B>,
    fc0: Linear<B, f32>,
    fc1: Linear<B, f32>,
    bn0: BatchNorm2d<B, f32>,
    bn1: BatchNorm2d<B, f32>,
    bn2: BatchNorm2d<B, f32>,
    bn3: BatchNorm1d<B, f32>,
}

impl<B: Backend> Discriminator<B> {
    pub fn new(features: usize, device: Device<B>) -> Result<Self> {
        let cv0 = Conv2D::new(1, features, 3, 3, 1, 1, None, true, true, device);
        let cv1 = Conv2D::new(3, features, 3, 3, 1, 1, None, true, true, device);
        let cv2 = Conv2D::new(features * 2, features, 3, 3, 1, 1, None, true, true, device);
        let down_conv_block0 = DiscriminatorDownConvBlock::new(features, features * 2, device);
        let down_conv_block1 = DiscriminatorDownConvBlock::new(features * 2, features * 4, device);
        let conv_block0 = DiscriminatorConvBlock::new(features * 4, device);
        let fc0 = Linear::new(128 * 8 * 8, 1024, true, device)?;
        let fc1 = Linear::new(1024, 1, true, device)?;
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        let bn1 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        let bn2 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        let bn3 = BatchNorm1d::new(1024, 0.9, 1e-4, device);
        let discriminator = Self {
            cv0,
            cv1,
            cv2,
            down_conv_block0,
            down_conv_block1,
            conv_block0,
            fc0,
            fc1,
            bn0,
            bn1,
            bn2,
            bn3,
        };
        Ok(discriminator)
    }

    pub fn forward(
        &mut self,
        x0: &Tensor<B, f32>,
        x1: &Tensor<B, f32>,
        is_train: bool,
    ) -> Result<Tensor<B, f32>> {
        let x0 = self.cv0.forward(x0)?;
        let x0 = self.bn0.forward(&x0, is_train)?;
        let x0 = x0.leaky_relu(0.2);

        let x1 = self.cv1.forward(x1)?;
        let x1 = self.bn1.forward(&x1, is_train)?;
        let x1 = x1.leaky_relu(0.2);

        let x = Tensor::cat(&vec![x0, x1], 1)?;
        let x = self.cv2.forward(&x)?;
        let x = self.bn2.forward(&x, is_train)?;
        let x = x.leaky_relu(0.2);

        let x = self.down_conv_block0.forward(&x, is_train)?;
        let x = self.down_conv_block1.forward(&x, is_train)?;
        let x = self.conv_block0.forward(&x, is_train)?;

        let x = flatten_except_batch(&x)?;
        let x = self.fc0.forward(&x)?;
        let x = self.bn3.forward(&x, is_train)?;
        let x = x.leaky_relu(0.2);

        let x = self.fc1.forward(&x)?;
        Ok(x)
    }
}

impl<B: Backend> Layer<B, f32> for Discriminator<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("cv1".to_string(), &self.cv1);
        map.insert("cv2".to_string(), &self.cv2);
        map.insert("down_conv_block0".to_string(), &self.down_conv_block0);
        map.insert("down_conv_block1".to_string(), &self.down_conv_block1);
        map.insert("conv_block0".to_string(), &self.conv_block0);
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map.insert("bn0".to_string(), &self.bn0);
        map.insert("bn1".to_string(), &self.bn1);
        map.insert("bn2".to_string(), &self.bn2);
        map.insert("bn3".to_string(), &self.bn3);
        map
    }
}

fn train<B: Backend>(device: Device<B>) -> Result<()> {
    let mut generator = Generator::new(32, device)?;
    let mut discriminator = Discriminator::new(32, device)?;
    let mut opt_gen = Adam::new(0.0002, 0.5, 0.999, 1e-7);
    let mut opt_dis = Adam::new(0.0002, 0.5, 0.999, 1e-7);

    let train_dataset = CIFAR10Loader::new("../datasets").load_train()?;
    let batch_size = 100;
    let max_epochs = 1;

    let label_real = Tensor::ones(vec![batch_size, 1], device);
    let label_fake = Tensor::zeros(vec![batch_size, 1], device);

    for epoch in 0..max_epochs {
        println!("epoch = {}", epoch);

        for (iter, (imgs, _)) in batch_iter(&train_dataset, batch_size, true, None).enumerate() {
            let imgs = imgs.to_device(device)?.to_dtype::<f32>()?;
            let imgs = ((imgs / ten![127.5].to_device(device)?)? - ten![1.0].to_device(device)?)?;

            let r = imgs.get_item(vec![(0, imgs.shape()[0]), (0, 1), (0, 32), (0, 32)])?;
            let g = imgs.get_item(vec![(0, imgs.shape()[0]), (1, 2), (0, 32), (0, 32)])?;
            let b = imgs.get_item(vec![(0, imgs.shape()[0]), (2, 3), (0, 32), (0, 32)])?;
            let grayscale_imgs = ((r + g + b)? / ten![3.0].to_device(device)?)?;

            let fake = generator.forward(&grayscale_imgs, true)?;
            let y_real = discriminator.forward(&grayscale_imgs, &imgs, true)?;
            let y_fake = discriminator.forward(&grayscale_imgs, &fake.detach(), true)?;
            let loss_dis_real = sigmoid_cross_entropy(&y_real, &label_real)?;
            let loss_dis_fake = sigmoid_cross_entropy(&y_fake, &label_fake)?;
            let loss_dis = (loss_dis_real + loss_dis_fake)?;
            let grads = loss_dis.backward()?;
            opt_dis.update_parameters(&mut discriminator.all_trainable_parameters_map(), &grads)?;

            let y_fake = discriminator.forward(&grayscale_imgs, &fake, true)?;
            let loss_gen = sigmoid_cross_entropy(&y_fake, &label_real)?;
            let grads = loss_gen.backward()?;
            opt_gen.update_parameters(&mut generator.all_trainable_parameters_map(), &grads)?;

            println!(
                "iter = {}, loss_dis = {}, loss_gen = {}",
                iter,
                loss_dis.to_vec()[0],
                loss_gen.to_vec()[0]
            );
        }

        let generator_safetensors = serialize(generator.all_parameters_map())?;
        fs::write(
            format!("../exclude/pix2pix_epoch{}.safetensors", epoch),
            generator_safetensors,
        )
        .unwrap();
    }

    Ok(())
}

fn test<B: Backend>(epoch: usize, device: Device<B>) -> Result<()> {
    let mut model = Generator::new(32, device)?;
    let model_safetensors =
        fs::read(format!("../exclude/pix2pix_epoch{}.safetensors", epoch)).unwrap();
    model.load_parameters_map(deserialize::<B, f32>(model_safetensors, device)?)?;

    let test_dataset = CIFAR10Loader::new("../datasets").load_test()?;

    let (imgs, _) = batch_iter(&test_dataset, 100, true, None).next().unwrap();
    let imgs = imgs.to_device(device)?.to_dtype::<f32>()?;
    let imgs = ((imgs / ten![127.5].to_device(device)?)? - ten![1.0].to_device(device)?)?;
    let r = imgs.get_item(vec![(0, imgs.shape()[0]), (0, 1), (0, 32), (0, 32)])?;
    let g = imgs.get_item(vec![(0, imgs.shape()[0]), (1, 2), (0, 32), (0, 32)])?;
    let b = imgs.get_item(vec![(0, imgs.shape()[0]), (2, 3), (0, 32), (0, 32)])?;
    let grayscale_imgs = ((r + g + b)? / ten![3.0].to_device(device)?)?;
    let color_imgs = model.forward(&grayscale_imgs, false)?;
    let result = ((color_imgs + 1.0)? * 127.5)?;
    let result = result.reshape(vec![10, 10, 3, 32, 32])?;
    let result = result
        .permuted_axes(&[0, 3, 1, 4, 2])?
        .reshape(vec![320, 320, 3])?;

    let mut pixel_data = Vec::new();
    for value in result.to_vec() {
        pixel_data.push(value as u8);
    }

    let img: RgbImage = ImageBuffer::from_raw(320, 320, pixel_data).expect("Invalid image buffer");
    img.save("../exclude/img.png").expect("Failed to save PNG");

    Ok(())
}

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    train(device)?;
    test(0, device)?;
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
