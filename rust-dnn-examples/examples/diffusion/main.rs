use std::{collections::HashMap, fs};

use image::{ImageBuffer, RgbImage};
use rust_dnn_core::{
    backend::Backend,
    device::{self, Device},
    tensor::Tensor,
};
use rust_dnn_datasets::mnist::MNISTLoader;
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::batch_iter,
    embedding::Embedding,
    layer::{BatchNorm2d, Conv2D, Deconv2D, Layer, Linear},
    loss::mean_squared_error,
    optimizer::{Adam, Optimizer},
};
use rust_dnn_safetensors::{deserialize, serialize};

fn pos_encoding<B: Backend>(timesteps: &Tensor<B, u32>, output_dim: usize) -> Tensor<B, f32> {
    let batch_size = timesteps.shape()[0];
    let v = Tensor::zeros(vec![batch_size, output_dim], timesteps.device());
    for i in 0..batch_size {
        let time_idx = timesteps.get_item(vec![(i, i + 1)]).to_vec()[0];
        let x = pos_encoding_time_idx(time_idx, output_dim, timesteps.device());
        v.set_item(&vec![(i, i + 1), (0, output_dim)], &x);
    }
    v
}

fn pos_encoding_time_idx<B: Backend>(
    time_idx: u32,
    output_dim: usize,
    device: Device<B>,
) -> Tensor<B, f32> {
    let mut v = Vec::new();
    for i in 0..output_dim {
        let div_term = (i as f32 / output_dim as f32 * (10000.0f32).ln()).exp();
        if i % 2 == 0 {
            v.push((time_idx as f32 / div_term).sin());
        } else {
            v.push((time_idx as f32 / div_term).cos());
        }
    }
    Tensor::from_vec(v, vec![output_dim], device)
}

struct TimeEmbedMlp<B: Backend> {
    fc0: Linear<B, f32>,
    fc1: Linear<B, f32>,
}

impl<B: Backend> TimeEmbedMlp<B> {
    pub fn new(in_size: usize, time_embed_dim: usize, device: Device<B>) -> Self {
        let fc0 = Linear::new(time_embed_dim, in_size, true, device);
        let fc1 = Linear::new(in_size, in_size, true, device);
        Self { fc0, fc1 }
    }

    pub fn forward(&self, x: &Tensor<B, f32>) -> Tensor<B, f32> {
        let x = self.fc0.forward(x);
        let x = x.relu();
        let x = self.fc1.forward(&x);
        x
    }
}

impl<B: Backend> Layer<B, f32> for TimeEmbedMlp<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map
    }
}

struct UpConvBlock<B: Backend> {
    mlp: TimeEmbedMlp<B>,
    dcv0: Deconv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> UpConvBlock<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        time_embed_dim: usize,
        device: Device<B>,
    ) -> Self {
        let mlp = TimeEmbedMlp::new(in_features, time_embed_dim, device);
        let dcv0 = Deconv2D::new(
            in_features,
            out_features,
            4,
            4,
            2,
            2,
            None,
            true,
            true,
            device,
        );
        let bn0 = BatchNorm2d::new(out_features, 0.9, 1e-4, device);
        Self { mlp, dcv0, bn0 }
    }

    pub fn forward(
        &mut self,
        x: &Tensor<B, f32>,
        v: &Tensor<B, f32>,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let v = self.mlp.forward(v);
        let v = v.reshape(vec![x.shape()[0], x.shape()[1], 1, 1]);
        let x = self.dcv0.forward(&(x + v));
        let x = self.bn0.forward(&x, is_train);
        let x = x.relu();
        x
    }
}

impl<B: Backend> Layer<B, f32> for UpConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("mlp".to_string(), &self.mlp);
        map.insert("dcv0".to_string(), &self.dcv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct DownConvBlock<B: Backend> {
    mlp: TimeEmbedMlp<B>,
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> DownConvBlock<B> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        time_embed_dim: usize,
        device: Device<B>,
    ) -> Self {
        let mlp = TimeEmbedMlp::new(in_features, time_embed_dim, device);
        let cv0 = Conv2D::new(
            in_features,
            out_features,
            4,
            4,
            2,
            2,
            None,
            true,
            true,
            device,
        );
        let bn0 = BatchNorm2d::new(out_features, 0.9, 1e-4, device);
        Self { mlp, cv0, bn0 }
    }

    pub fn forward(
        &mut self,
        x: &Tensor<B, f32>,
        v: &Tensor<B, f32>,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let v = self.mlp.forward(&v);
        let v = v.reshape(vec![x.shape()[0], x.shape()[1], 1, 1]);
        let x = self.cv0.forward(&(x + v));
        let x = self.bn0.forward(&x, is_train);
        let x = x.relu();
        x
    }
}

impl<B: Backend> Layer<B, f32> for DownConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("mlp".to_string(), &self.mlp);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct ConvBlock<B: Backend> {
    mlp: TimeEmbedMlp<B>,
    cv0: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(features: usize, time_embed_dim: usize, device: Device<B>) -> Self {
        let mlp = TimeEmbedMlp::new(features, time_embed_dim, device);
        let cv0 = Conv2D::new(features, features, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        Self { mlp, cv0, bn0 }
    }

    pub fn forward(
        &mut self,
        x: &Tensor<B, f32>,
        v: &Tensor<B, f32>,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let v = self.mlp.forward(&v);
        let v = v.reshape(vec![x.shape()[0], x.shape()[1], 1, 1]);
        let x = self.cv0.forward(&(x + v));
        let x = self.bn0.forward(&x, is_train);
        let x = x.relu();
        x
    }
}

impl<B: Backend> Layer<B, f32> for ConvBlock<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("mlp".to_string(), &self.mlp);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct UNet<B: Backend> {
    time_embed_dim: usize,
    mlp0: TimeEmbedMlp<B>,
    emb0: Embedding<B, f32>,
    down_conv_block0: DownConvBlock<B>,
    down_conv_block1: DownConvBlock<B>,
    conv_block0: ConvBlock<B>,
    up_conv_block0: UpConvBlock<B>,
    up_conv_block1: UpConvBlock<B>,
    cv0: Conv2D<B, f32>,
    cv1: Conv2D<B, f32>,
    bn0: BatchNorm2d<B, f32>,
}

impl<B: Backend> UNet<B> {
    pub fn new(
        features: usize,
        num_labels: usize,
        time_embed_dim: usize,
        device: Device<B>,
    ) -> Self {
        let down_conv_block0 = DownConvBlock::new(features, features * 2, time_embed_dim, device);
        let down_conv_block1 =
            DownConvBlock::new(features * 2, features * 4, time_embed_dim, device);
        let conv_block0 = ConvBlock::new(features * 4, time_embed_dim, device);
        let up_conv_block0 = UpConvBlock::new(
            features * 4 + features * 4,
            features * 2,
            time_embed_dim,
            device,
        );
        let up_conv_block1 = UpConvBlock::new(
            features * 2 + features * 2,
            features,
            time_embed_dim,
            device,
        );
        let in_ch = 1;
        let mlp0 = TimeEmbedMlp::new(in_ch, time_embed_dim, device);
        let emb0 = Embedding::new(num_labels, time_embed_dim, device).unwrap();
        let cv0 = Conv2D::new(1, features, 3, 3, 1, 1, None, true, true, device);
        let cv1 = Conv2D::new(features, 1, 3, 3, 1, 1, None, true, true, device);
        let bn0 = BatchNorm2d::new(features, 0.9, 1e-4, device);
        Self {
            time_embed_dim,
            mlp0,
            emb0,
            down_conv_block0,
            down_conv_block1,
            conv_block0,
            up_conv_block0,
            up_conv_block1,
            cv0,
            cv1,
            bn0,
        }
    }

    pub fn forward(
        &mut self,
        x: &Tensor<B, f32>,
        timesteps: &Tensor<B, u32>,
        labels: &Tensor<B, u32>,
        is_train: bool,
    ) -> Tensor<B, f32> {
        let v = pos_encoding(timesteps, self.time_embed_dim);
        let v_labels = self.emb0.forward(&labels);
        let v = v + v_labels;

        let v2 = self.mlp0.forward(&v);
        let v2 = v2.reshape(vec![x.shape()[0], x.shape()[1], 1, 1]);

        let x = self.cv0.forward(&(x + &v2));
        let x = self.bn0.forward(&x, is_train);
        let x = x.relu();

        let x = self.down_conv_block0.forward(&x, &v, is_train);
        let x0 = &x;
        let x = self.down_conv_block1.forward(&x, &v, is_train);
        let x1 = &x;
        let x = self.conv_block0.forward(&x, &v, is_train);
        let x = Tensor::cat(&vec![x1.clone(), x.clone()], 1);
        let x = self.up_conv_block0.forward(&x, &v, is_train);
        let x = Tensor::cat(&vec![x0.clone(), x.clone()], 1);
        let x = self.up_conv_block1.forward(&x, &v, is_train);

        let x = self.cv1.forward(&x);
        x
    }
}

impl<B: Backend> Layer<B, f32> for UNet<B> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, f32>> {
        let mut map: HashMap<String, &dyn Layer<B, f32>> = HashMap::new();
        map.insert("down_conv_block0".to_string(), &self.down_conv_block0);
        map.insert("down_conv_block1".to_string(), &self.down_conv_block1);
        map.insert("conv_block0".to_string(), &self.conv_block0);
        map.insert("up_conv_block0".to_string(), &self.up_conv_block0);
        map.insert("up_conv_block1".to_string(), &self.up_conv_block1);
        map.insert("emb0".to_string(), &self.emb0);
        map.insert("mlp0".to_string(), &self.mlp0);
        map.insert("cv0".to_string(), &self.cv0);
        map.insert("cv1".to_string(), &self.cv1);
        map.insert("bn0".to_string(), &self.bn0);
        map
    }
}

struct Diffuser<B: Backend> {
    num_timesteps: usize,

    alphas: Tensor<B, f32>,
    alpha_bars: Tensor<B, f32>,
}

impl<B: Backend> Diffuser<B> {
    pub fn new(num_timesteps: usize, beta_start: f32, beta_end: f32, device: Device<B>) -> Self {
        let betas = Tensor::linspace(beta_start, beta_end, num_timesteps, device);
        let alphas = 1.0 - &betas;
        let alpha_bars = alphas.cumprod(0);
        Self {
            num_timesteps,
            alphas,
            alpha_bars,
        }
    }

    pub fn add_noise(
        &self,
        x_0: &Tensor<B, f32>,
        t: &Tensor<B, u32>,
    ) -> (Tensor<B, f32>, Tensor<B, f32>) {
        let t_idx = t - 1.0;
        let alpha_bar = self.alpha_bars.gather(&t_idx, 0);
        let n = alpha_bar.len();
        let alpha_bar = alpha_bar.reshape(vec![n, 1, 1, 1]);
        let noise = Tensor::rand_norm(x_0.shape().clone(), None, x_0.device());
        let x_t = alpha_bar.sqrt() * x_0 + (1.0 - alpha_bar).sqrt() * &noise;
        (x_t, noise)
    }

    pub fn denoise(
        &self,
        model: &mut UNet<B>,
        x: &Tensor<B, f32>,
        t: &Tensor<B, u32>,
        labels: &Tensor<B, u32>,
    ) -> Tensor<B, f32> {
        let t_idx = t - 1.0;
        let alpha = self.alphas.gather(&t_idx.clone(), 0);
        let alpha_bar = self.alpha_bars.gather(&t_idx.clone(), 0);
        let alpha_bar_prev = self.alpha_bars.gather(&(t_idx - 1.0), 0);

        let n = alpha_bar.len();
        let alpha = alpha.reshape(vec![n, 1, 1, 1]);
        let alpha_bar = alpha_bar.reshape(vec![n, 1, 1, 1]);
        let alpha_bar_prev = alpha_bar_prev.reshape(vec![n, 1, 1, 1]);

        // let is_test_prev = begin_test();
        let eps = model.forward(x, t, labels, false);
        // end_test(is_test_prev);

        let noise = Tensor::rand_norm(x.shape(), None, x.device());
        let t_not_one_mask = -t.eq_scalar(1.0).to_dtype::<f32>() + 1.0; // TODO: not_eqを作りたい
        let noise = noise * t_not_one_mask.reshape(vec![n, 1, 1, 1]); // no noise at t=1

        let mu = (x - ((1.0 - &alpha) / (1.0 - &alpha_bar).sqrt()) * eps) / &alpha.sqrt();
        let std = ((1.0 - &alpha) * (1.0 - &alpha_bar_prev) / (1.0 - &alpha_bar)).sqrt();
        mu + noise * std
    }

    pub fn sample(
        &self,
        model: &mut UNet<B>,
        num_labels: usize,
        x_shape: Vec<usize>,
        device: Device<B>,
    ) -> Tensor<B, f32> {
        let batch_size = x_shape[0];

        let labels = Tensor::zeros(vec![batch_size], device);
        for i in 0..batch_size {
            let label = Tensor::from_scalar((i % num_labels) as u32, device);
            labels.set_item(&vec![(i, i + 1)], &label);
        }

        let mut x = Tensor::rand_norm(&x_shape, None, device);

        // TODO: なぜか2..self.num_timestepsの範囲にしないとdenoise内でgatherに渡るindexがアンダーフローしてしまう。
        for i in (2..self.num_timesteps).rev() {
            println!("i = {}", i);
            let t = Tensor::fill(vec![batch_size], i as u32, device);
            x = self.denoise(model, &x, &t, &labels);
            // TODO: これがないとなぜか計算グラフが作られてしまう。enable_backpropとis_testを分けているのが原因?
            x = x.detach();
        }
        x
    }
}

fn train<B: Backend>(device: Device<B>) {
    let num_timesteps = 1000;
    let beta_start = 0.0001;
    let beta_end = 0.02;
    let diffuser = Diffuser::new(num_timesteps, beta_start, beta_end, device);
    let mut model = UNet::new(64, 10, 100, device);
    let mut optimizer = Adam::default();

    let batch_size = 100;
    let max_epochs = 1;

    let train_dataset = MNISTLoader::new("../datasets").load_train().unwrap();

    for epoch in 0..max_epochs {
        println!("epoch = {}", epoch);
        for (iter, (images, labels)) in
            batch_iter(&train_dataset, batch_size, true, None).enumerate()
        {
            let images = images.to_device(device).unwrap().to_dtype::<f32>();
            let images = images.reshape(vec![batch_size, 1, 28, 28]);
            let images = (images / 127.5) - 1.0;

            let labels = labels.to_device(device).unwrap();
            let labels = labels.reshape(vec![labels.shape()[0]]);

            let t = Tensor::rand_int(1, num_timesteps as i32 + 1, &[batch_size], None, device);
            let (x_noisy, noise) = diffuser.add_noise(&images, &t);
            let y = model.forward(&x_noisy, &t, &labels, true);
            let loss = mean_squared_error(&y, &noise).unwrap();
            let grads = loss.backward();
            println!("iter = {}, loss = {}", iter, loss.to_vec()[0]);
            optimizer.update_parameters(&mut model.all_trainable_parameters_map(), &grads);
        }

        let model_safetensors = serialize(model.all_parameters_map()).unwrap();
        let path = format!("../exclude/ddpm_epoch{}.safetensors", epoch);
        fs::write(path, model_safetensors).unwrap();
    }
}

fn test<B: Backend>(epoch: usize, device: Device<B>) {
    let num_timesteps = 1000;
    let beta_start = 0.0001;
    let beta_end = 0.02;
    let diffuser = Diffuser::new(num_timesteps, beta_start, beta_end, device);
    let mut model = UNet::new(64, 10, 100, device);

    let path = format!("../exclude/ddpm_epoch{}.safetensors", epoch);
    let model_safetensors = fs::read(path).unwrap();
    model
        .load_parameters_map(deserialize(model_safetensors, device).unwrap())
        .unwrap();

    // set_test(true);

    let y = diffuser.sample(&mut model, 10, vec![100, 1, 28, 28], device);
    let result = (y + 1.0) * 127.5;
    let result = Tensor::cat(&vec![result.clone(), result.clone(), result.clone()], 1);
    let result = result.reshape(vec![10, 10, 3, 28, 28]);
    let result = result
        .permuted_axes(&[0, 3, 1, 4, 2])
        .reshape(vec![280, 280, 3]);

    let mut pixel_data = Vec::new();
    for value in result.to_vec() {
        let value = value.clamp(0.0, 255.0);
        pixel_data.push(value as u8);
    }

    let img: RgbImage = ImageBuffer::from_raw(280, 280, pixel_data).expect("Invalid image buffer");
    img.save("../exclude/img.png").expect("Failed to save PNG");
}

fn run<B: Backend>(device: Device<B>) {
    train(device);
    test(0, device);
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
