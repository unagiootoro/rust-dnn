use std::{collections::HashMap, marker::PhantomData};

use rust_dnn_core::{backend::Backend, float::Float, tensor::Tensor};
use rust_dnn_nn::layer::{Conv2D, Layer};

use crate::{unet_attention_block::UNET_AttentionBlock, unet_residual_block::UNET_ResidualBlock, upsampling::Upsample};

pub trait SwitchSequentialItem<B: Backend, T: Float> {
    fn forward(
        &self,
        x: &Tensor<B, T>,
        context: &Tensor<B, T>,
        time: &Tensor<B, T>,
    ) -> Tensor<B, T>;
}

pub trait SwitchSequentialLayer<B: Backend, T: Float>:
    Layer<B, T> + SwitchSequentialItem<B, T>
{
    fn as_layer(&self) -> &dyn Layer<B, T>;
}

impl<L, B, T> SwitchSequentialLayer<B, T> for L
where
    L: Layer<B, T> + SwitchSequentialItem<B, T>,
    B: Backend,
    T: Float,
{
    fn as_layer(&self) -> &dyn Layer<B, T> {
        self
    }
}

pub struct SwitchSequential<B, T>
where
    B: Backend,
    T: Float,
{
    vec: Vec<Box<dyn SwitchSequentialLayer<B, T>>>,
    device_marker: PhantomData<B>,
    dtype_marker: PhantomData<T>,
}

impl<B, T> SwitchSequential<B, T>
where
    B: Backend,
    T: Float,
{
    pub fn new() -> Self {
        Self {
            vec: Vec::new(),
            device_marker: PhantomData,
            dtype_marker: PhantomData,
        }
    }

    pub fn from_vec(vec: Vec<Box<dyn SwitchSequentialLayer<B, T>>>) -> Self {
        Self {
            vec,
            device_marker: PhantomData,
            dtype_marker: PhantomData,
        }
    }

    pub fn add<L: Layer<B, T> + SwitchSequentialItem<B, T> + 'static>(&mut self, layer: L) {
        self.vec.push(Box::new(layer));
    }

    pub fn get(&self, i: usize) -> &dyn SwitchSequentialLayer<B, T> {
        &*self.vec[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut dyn SwitchSequentialLayer<B, T> {
        &mut *self.vec[i]
    }
}

impl<B, T> SwitchSequentialItem<B, T> for SwitchSequential<B, T>
where
    B: Backend,
    T: Float,
{
    fn forward(
        &self,
        x: &Tensor<B, T>,
        context: &Tensor<B, T>,
        time: &Tensor<B, T>,
    ) -> Tensor<B, T> {
        let mut x = x.clone();
        for layer in &self.vec {
            x = layer.forward(&x, context, time);
        }
        x
    }
}

impl<B, T> Layer<B, T> for SwitchSequential<B, T>
where
    B: Backend,
    T: Float,
{
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map = HashMap::<String, &dyn Layer<B, T>>::new();
        for (i, layer) in self.vec.iter().enumerate() {
            // as_layer() を経由して &dyn Layer を取得
            map.insert(i.to_string(), layer.as_layer());
        }
        map
    }

    fn parameters_map(
        &self,
    ) -> std::collections::HashMap<String, rust_dnn_core::tensor::Tensor<B, T>> {
        std::collections::HashMap::new()
    }

    fn trainable_parameters_map(
        &self,
    ) -> std::collections::HashMap<String, rust_dnn_core::tensor::Tensor<B, T>> {
        self.parameters_map()
    }

    fn all_parameters_map(
        &self,
    ) -> std::collections::HashMap<String, rust_dnn_core::tensor::Tensor<B, T>> {
        let mut map = std::collections::HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_parameters_map() {
                let name = std::format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn all_trainable_parameters_map(
        &self,
    ) -> std::collections::HashMap<String, rust_dnn_core::tensor::Tensor<B, T>> {
        let mut map = std::collections::HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_trainable_parameters_map() {
                let name = std::format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.trainable_parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn load_parameters_map(
        &mut self,
        map: std::collections::HashMap<String, rust_dnn_core::tensor::Tensor<B, T>>,
    ) -> rust_dnn_core::error::Result<()> {
        for (name, parameter) in self.all_parameters_map() {
            if let Some(param) = map.get(&name) {
                parameter.copy(param);
            }
        }
        Ok(())
    }
}

impl<B: Backend, T: Float> SwitchSequentialItem<B, T> for Conv2D<B, T> {
    fn forward(
        &self,
        x: &Tensor<B, T>,
        _context: &Tensor<B, T>,
        _time: &Tensor<B, T>,
    ) -> Tensor<B, T> {
        Conv2D::forward(&self, x)
    }
}

impl<B: Backend> SwitchSequentialItem<B, f32> for Upsample<B> {
    fn forward(
        &self,
        x: &Tensor<B, f32>,
        _context: &Tensor<B, f32>,
        _time: &Tensor<B, f32>,
    ) -> Tensor<B, f32> {
        Upsample::forward(&self, x)
    }
}

impl<B: Backend> SwitchSequentialItem<B, f32> for UNET_AttentionBlock<B> {
    fn forward(
        &self,
        x: &Tensor<B, f32>,
        context: &Tensor<B, f32>,
        _time: &Tensor<B, f32>,
    ) -> Tensor<B, f32> {
        UNET_AttentionBlock::forward(&self, x, context)
    }
}

impl<B: Backend> SwitchSequentialItem<B, f32> for UNET_ResidualBlock<B> {
    fn forward(
        &self,
        x: &Tensor<B, f32>,
        _context: &Tensor<B, f32>,
        time: &Tensor<B, f32>,
    ) -> Tensor<B, f32> {
        UNET_ResidualBlock::forward(&self, x, time)
    }
}
