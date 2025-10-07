use std::collections::HashMap;

use crate::{backend::Backend, error::Error, num::Num, tensor::Tensor};

pub struct Gradients<B: Backend, T: Num>(HashMap<usize, Tensor<B, T>>);

impl<B: Backend, T: Num> Gradients<B, T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get_by_id(&self, id: usize) -> Option<&Tensor<B, T>> {
        self.0.get(&id)
    }

    pub fn get(&self, tensor: &Tensor<B, T>) -> Option<&Tensor<B, T>> {
        self.0.get(&tensor.id())
    }

    pub fn insert(&mut self, tensor: &Tensor<B, T>, grad: Tensor<B, T>) {
        self.0.insert(tensor.id(), grad);
    }

    pub fn add(&mut self, tensor: &Tensor<B, T>, grad: Tensor<B, T>) -> Result<(), Error> {
        if let Some(prev_grad) = self.0.get(&tensor.id()) {
            self.0.insert(tensor.id(), (prev_grad.clone() + grad)?);
        } else {
            self.0.insert(tensor.id(), grad);
        }
        Ok(())
    }

    pub fn remove(&mut self, tensor: &Tensor<B, T>) {
        self.0.remove(&tensor.id());
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}
