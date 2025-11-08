use std::collections::HashMap;

use rust_dnn_core::backend::Backend;
use rust_dnn_core::device::Device;
use rust_dnn_core::error::Result;
use rust_dnn_core::float::Float;
use rust_dnn_core::tensor::Tensor;

pub trait Layer<B: Backend, T: Float> {
    fn layers_map(&self) -> HashMap<String, &dyn Layer<B, T>> {
        HashMap::new()
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        HashMap::new()
    }

    fn trainable_parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        self.parameters_map()
    }

    fn all_parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_parameters_map() {
                let name = format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn all_trainable_parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        for (layer_name, layer) in self.layers_map() {
            for (parameter_name, parameter) in layer.all_trainable_parameters_map() {
                let name = format!("{}.{}", layer_name, parameter_name);
                map.insert(name, parameter);
            }
        }
        for (parameter_name, parameter) in self.trainable_parameters_map() {
            map.insert(parameter_name, parameter);
        }
        map
    }

    fn load_parameters_map(&mut self, map: HashMap<String, Tensor<B, T>>) -> Result<()> {
        for (name, parameter) in self.all_parameters_map() {
            if let Some(param) = map.get(&name) {
                parameter.copy(param)?;
            }
        }
        Ok(())
    }
}

pub struct Linear<B: Backend, T: Float> {
    weight: Tensor<B, T>,
    bias: Option<Tensor<B, T>>,
}

impl<B: Backend, T: Float> Layer<B, T> for Linear<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::<String, Tensor<B, T>>::new();
        map.insert("weight".to_string(), self.weight.clone());
        if let Some(bias) = self.bias.as_ref() {
            map.insert("bias".to_string(), bias.clone());
        };
        map
    }
}

impl<B: Backend, T: Float> Linear<B, T> {
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        device: Device<B>,
    ) -> Result<Self> {
        let weight = (Tensor::rand_norm(&[out_features, in_features], None, device)
            * Tensor::from_scalar(T::from_f64(1.0 / in_features as f64), device))?
        .requires_grad();
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_features], device).requires_grad())
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    pub fn from_weights(weight: Tensor<B, T>, bias: Option<Tensor<B, T>>) -> Result<Self> {
        Ok(Self { weight, bias })
    }

    pub fn weight(&self) -> &Tensor<B, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<B, T>> {
        self.bias.as_ref()
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        linear::<B, T>(x, &self.weight, self.bias.as_ref())
    }
}

pub fn linear<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    weight: &Tensor<B, T>,
    bias: Option<&Tensor<B, T>>,
) -> Result<Tensor<B, T>> {
    if let Some(bias) = bias {
        x.matmul(&weight.reversed_axes()?) + bias
    } else {
        x.matmul(&weight.reversed_axes()?)
    }
}
