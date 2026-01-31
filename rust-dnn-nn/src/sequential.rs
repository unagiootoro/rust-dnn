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

// Layer と SequentialItem の両方を満たすトレイトオブジェクト用の中継トレイト
// Rust の制限により、Box<dyn Layer + SequentialItem> とは書けないため、このように定義します。
pub trait SequentialLayer<B: Backend, T: Float>: Layer<B, T> + SequentialItem<B, T> {
    // トレイトオブジェクトから Layer トレイトへの参照を取得するためのヘルパー
    fn as_layer(&self) -> &dyn Layer<B, T>;
}

// 既存のすべてのレイヤーに対して自動的に SequentialLayer を実装する設定
impl<L, B, T> SequentialLayer<B, T> for L
where
    L: Layer<B, T> + SequentialItem<B, T>,
    B: Backend,
    T: Float,
{
    fn as_layer(&self) -> &dyn Layer<B, T> {
        self
    }
}

pub struct DynSequential<B, T>
where
    B: Backend,
    T: Float,
{
    // L を削除し、中継トレイトのトレイトオブジェクトを保持
    vec: Vec<Box<dyn SequentialLayer<B, T>>>,
    device_marker: PhantomData<B>,
    dtype_marker: PhantomData<T>,
}

impl<B, T> DynSequential<B, T>
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

    pub fn from_vec(vec: Vec<Box<dyn SequentialLayer<B, T>>>) -> Self {
        Self {
            vec,
            device_marker: PhantomData,
            dtype_marker: PhantomData,
        }
    }

    // Layer と SequentialItem の両方を実装している型なら何でも追加可能
    pub fn add<L: Layer<B, T> + SequentialItem<B, T> + 'static>(&mut self, layer: L) {
        self.vec.push(Box::new(layer));
    }

    pub fn get(&self, i: usize) -> &dyn SequentialLayer<B, T> {
        &*self.vec[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut dyn SequentialLayer<B, T> {
        &mut *self.vec[i]
    }
}

impl<B, T> SequentialItem<B, T> for DynSequential<B, T>
where
    B: Backend,
    T: Float,
{
    fn forward(&mut self, x: Tensor<B, T>, is_train: bool) -> Tensor<B, T> {
        let mut x = x;
        for layer in &mut self.vec {
            x = layer.forward(x, is_train);
        }
        x
    }
}

impl<B, T> Layer<B, T> for DynSequential<B, T>
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
