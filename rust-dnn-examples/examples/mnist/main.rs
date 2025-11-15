use std::collections::{HashMap, VecDeque};

use rand::prelude::*;
use rust_dnn_core::{
    backend::Backend, device::Device, error::Result, float::Float, num::Num, ten, tensor::Tensor,
};
use rust_dnn_datasets::mnist::MNISTLoader;
use rust_dnn_examples::argv::get_argv;
use rust_dnn_nn::{
    batch_iter::batch_iter,
    layer::{Layer, Linear},
    loss::cross_entropy,
    optimizer::{Adam, Optimizer, SGD},
};

struct Model<B: Backend, T: Float> {
    fc0: Linear<B, T>,
    fc1: Linear<B, T>,
    fc2: Linear<B, T>,
}

impl<B: Backend, T: Float> Model<B, T> {
    pub fn new(device: Device<B>) -> Result<Self> {
        let fc0 = Linear::new(784, 256, true, device)?;
        let fc1 = Linear::new(256, 256, true, device)?;
        let fc2 = Linear::new(256, 10, true, device)?;
        Ok(Self { fc0, fc1, fc2 })
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Model<B, T> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map: HashMap<String, &dyn Layer<B, T>> = HashMap::new();
        map.insert("fc0".to_string(), &self.fc0);
        map.insert("fc1".to_string(), &self.fc1);
        map.insert("fc2".to_string(), &self.fc2);
        map
    }
}

impl<B: Backend, T: Float> Model<B, T> {
    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        let x = self.fc0.forward(x)?;
        let x = x.relu();
        let x = self.fc1.forward(&x)?;
        let x = x.relu();
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

fn accuracy<B: Backend, T: Float>(x: &Tensor<B, T>, t: &Tensor<B, u32>) -> Result<usize> {
    let correct = x.argmax_axis(1, false)?.eq(t)?.to_dtype::<T>()?.sum()?;
    Ok(correct.to_vec()[0].as_usize())
}

fn run<B: Backend>(device: Device<B>) -> Result<()> {
    let train_dataset = MNISTLoader::new("../datasets").load_train()?;
    let test_dataset = MNISTLoader::new("../datasets").load_test()?;

    let model = Model::<_, f32>::new(device)?;
    let mut optimizer = Adam::default();

    let batch_size = 100;
    let max_epochs = 1;

    {
        for epoch in 0..max_epochs {
            println!("epoch = {}", epoch);
            for (iter, (images, labels)) in
                batch_iter(&train_dataset, batch_size, true, None).enumerate()
            {
                let images = images.to_device(device)?.to_dtype::<f32>()?;
                let images = (images / ten![255.0].to_device(device)?)?;
                let images = images.reshape(vec![100, 784])?;
                let y = model.forward(&images)?;
                let labels = labels.to_device(device)?;
                let loss = cross_entropy(&y, &labels)?;
                let grads = loss.backward()?;
                println!("iter = {}, loss = {}", iter, loss.to_vec()[0]);
                optimizer.update_parameters(&mut model.all_trainable_parameters_map(), &grads)?;
            }
        }
    }

    {
        let mut correct = 0;
        for (images, labels) in batch_iter(&test_dataset, batch_size, false, None) {
            let images = images.to_device(device)?.to_dtype::<f32>()?;
            let images = (images / ten![255.0].to_device(device)?)?;
            let images = images.reshape(vec![100, 784])?;
            let y = model.forward(&images)?;
            let labels = labels.to_device(device)?;
            correct += accuracy(&y, &labels)?;
        }

        println!("correct = {}", correct);
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
