use std::{collections::HashMap, marker::PhantomData};

use rust_dnn_core::{backend::Backend, float::Float, tensor::Tensor};

use crate::layer::Layer;

pub trait SequentialItem<B: Backend, T: Float> {
    fn forward(&mut self, x: Tensor<B, T>, is_train: bool) -> Tensor<B, T>;
}

pub struct Sequential<L, B, T>
where
    L: Layer<B, T> + SequentialItem<B, T>,
    B: Backend,
    T: Float,
{
    vec: Vec<L>,
    device_marker: PhantomData<B>,
    dtype_marker: PhantomData<T>,
}

impl<L, B, T> Sequential<L, B, T>
where
    L: Layer<B, T> + SequentialItem<B, T>,
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

    pub fn from_vec(vec: Vec<L>) -> Self {
        Self {
            vec,
            device_marker: PhantomData,
            dtype_marker: PhantomData,
        }
    }

    pub fn add(&mut self, layer: L) {
        self.vec.push(layer);
    }

    pub fn get(&self, i: usize) -> &L {
        &self.vec[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut L {
        &mut self.vec[i]
    }
}

impl<L, B, T> SequentialItem<B, T> for Sequential<L, B, T>
where
    L: Layer<B, T> + SequentialItem<B, T>,
    B: Backend,
    T: Float,
{
    fn forward(&mut self, x: Tensor<B, T>, is_train: bool) -> Tensor<B, T> {
        let mut x = x.clone();
        for layer in &mut self.vec {
            x = layer.forward(x, is_train);
        }
        x
    }
}

impl<L, B, T> Layer<B, T> for Sequential<L, B, T>
where
    L: Layer<B, T> + SequentialItem<B, T>,
    B: Backend,
    T: Float,
{
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        let mut map = HashMap::<String, &dyn Layer<B, T>>::new();
        for (i, layer) in self.vec.iter().enumerate() {
            map.insert(i.to_string(), &*layer);
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
