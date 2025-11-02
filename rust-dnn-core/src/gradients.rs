use std::collections::HashMap;

use crate::{
    backend::Backend,
    dim::{Dim, DynDim},
    error::Error,
    num::Num,
    tensor::Tensor,
};

pub struct Gradients<B: Backend, T: Num>(HashMap<usize, Tensor<B, DynDim, T>>);

impl<B: Backend, T: Num> Gradients<B, T> {
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    pub fn get_by_id(&self, id: usize) -> Option<&Tensor<B, DynDim, T>> {
        self.0.get(&id)
    }

    pub fn get<D: Dim>(&self, tensor: &Tensor<B, D, T>) -> Option<&Tensor<B, D, T>> {
        if let Some(t) = self.0.get(&tensor.id()) {
            Some(t.to_dim_ref::<D>().unwrap())
        } else {
            None
        }
    }

    pub fn insert(&mut self, tensor: &Tensor<B, DynDim, T>, grad: Tensor<B, DynDim, T>) {
        self.0.insert(tensor.id(), grad);
    }

    pub fn add(
        &mut self,
        tensor: &Tensor<B, DynDim, T>,
        grad: Tensor<B, DynDim, T>,
    ) -> Result<(), Error> {
        if let Some(prev_grad) = self.0.get(&tensor.id()) {
            self.0.insert(tensor.id(), (prev_grad.clone() + grad)?);
        } else {
            self.0.insert(tensor.id(), grad);
        }
        Ok(())
    }

    pub fn remove(&mut self, tensor: &Tensor<B, DynDim, T>) {
        self.0.remove(&tensor.id());
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}
