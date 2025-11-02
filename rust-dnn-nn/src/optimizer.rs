use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend, error::Result, float::Float, gradients::Gradients, tensor::Tensor,
};

pub trait Optimizer<B: Backend, T: Float> {
    fn update_parameters(
        &mut self,
        parameters: &mut HashMap<String, Tensor<B, T>>,
        grads: &Gradients<B, T>,
    ) -> Result<()> {
        self.prepare_update_parameters();
        for (name, parameter) in parameters {
            if let Some(grad) = grads.get(parameter) {
                self.update_parameter(name, parameter, &grad.detach())?;
            } else {
                panic!("{}: grad is None", name);
            }
        }
        Ok(())
    }

    fn prepare_update_parameters(&mut self) {}

    fn update_parameter(
        &mut self,
        name: &str,
        parameter: &mut Tensor<B, T>,
        grad: &Tensor<B, T>,
    ) -> Result<()>;

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>>;

    fn load_parameters_map(&mut self, map: HashMap<String, Tensor<B, T>>) -> Result<()> {
        for (name, parameter) in self.parameters_map() {
            parameter.copy(&map[&name])?;
        }
        Ok(())
    }
}

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }

    pub fn default() -> Self {
        Self::new(0.01)
    }
}

impl<B: Backend, T: Float> Optimizer<B, T> for SGD {
    fn update_parameter(
        &mut self,
        _name: &str,
        parameter: &mut Tensor<B, T>,
        grad: &Tensor<B, T>,
    ) -> Result<()> {
        let lr = Tensor::from_scalar(T::from_f64(self.lr), grad.device());
        parameter.sub_assign(&((lr * grad)?))
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        HashMap::new()
    }
}
