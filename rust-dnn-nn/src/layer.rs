use std::collections::HashMap;

use rust_dnn_core::backend::Backend;
use rust_dnn_core::device::Device;
use rust_dnn_core::error::Result;
use rust_dnn_core::float::Float;
use rust_dnn_core::num::Num;
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

pub struct Conv2D<B: Backend, T: Float> {
    in_filters: usize,
    out_filters: usize,
    fil_h: usize,
    fil_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding: Option<(usize, usize)>,
    auto_padding: bool,
    weight: Tensor<B, T>,
    bias: Option<Tensor<B, T>>,
}

impl<B: Backend, T: Float> Conv2D<B, T> {
    pub fn new(
        in_filters: usize,
        out_filters: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Option<(usize, usize)>,
        auto_padding: bool,
        use_bias: bool,
        device: Device<B>,
    ) -> Self {
        let scale = (1.0 / (fil_h * fil_w * in_filters) as f64).sqrt();
        let weight = (Tensor::rand_norm(&[out_filters, in_filters, fil_h, fil_w], None, device)
            * T::from_f64(scale))
        .unwrap();
        let weight = weight.requires_grad();
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_filters], device).requires_grad())
        } else {
            None
        };
        Self {
            in_filters,
            out_filters,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
            padding,
            auto_padding,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        x.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.in_filters,
            self.out_filters,
            self.fil_h,
            self.fil_w,
            self.stride_h,
            self.stride_w,
            self.padding,
            self.auto_padding,
        )
    }

    pub fn weight(&self) -> &Tensor<B, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<B, T>> {
        self.bias.as_ref()
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Conv2D<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::<String, Tensor<B, T>>::new();
        map.insert("weight".to_string(), self.weight.clone());
        if let Some(bias) = self.bias.as_ref() {
            map.insert("bias".to_string(), bias.clone());
        };
        map
    }
}

pub struct Deconv2D<B: Backend, T: Float> {
    in_filters: usize,
    out_filters: usize,
    fil_h: usize,
    fil_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding: Option<(usize, usize)>,
    auto_padding: bool,
    weight: Tensor<B, T>,
    bias: Option<Tensor<B, T>>,
}

impl<B: Backend, T: Float> Deconv2D<B, T> {
    pub fn new(
        in_filters: usize,
        out_filters: usize,
        fil_h: usize,
        fil_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding: Option<(usize, usize)>,
        auto_padding: bool,
        use_bias: bool,
        device: Device<B>,
    ) -> Self {
        let scale = (1.0 / (fil_h * fil_w * in_filters) as f64).sqrt();
        let weight = (Tensor::rand_norm(&[in_filters, out_filters, fil_h, fil_w], None, device)
            * T::from_f64(scale))
        .unwrap();
        let weight = weight.requires_grad();
        let bias = if use_bias {
            Some(Tensor::zeros(vec![out_filters], device).requires_grad())
        } else {
            None
        };
        Self {
            in_filters,
            out_filters,
            fil_h,
            fil_w,
            stride_h,
            stride_w,
            padding,
            auto_padding,
            weight,
            bias,
        }
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        x.deconv2d(
            &self.weight,
            self.bias.as_ref(),
            self.in_filters,
            self.out_filters,
            self.fil_h,
            self.fil_w,
            self.stride_h,
            self.stride_w,
            self.padding,
            self.auto_padding,
        )
    }

    pub fn weight(&self) -> &Tensor<B, T> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<B, T>> {
        self.bias.as_ref()
    }
}

impl<B: Backend, T: Float> Layer<B, T> for Deconv2D<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::<String, Tensor<B, T>>::new();
        map.insert("weight".to_string(), self.weight.clone());
        if let Some(bias) = self.bias.as_ref() {
            map.insert("bias".to_string(), bias.clone());
        };
        map
    }
}

pub struct BatchNorm1d<B: Backend, T: Float> {
    running_mean: Tensor<B, T>,
    running_var: Tensor<B, T>,
    gamma: Tensor<B, T>,
    beta: Tensor<B, T>,
    momentum: T,
    eps: T,
}

impl<B, T> BatchNorm1d<B, T>
where
    B: Backend,
    T: Float,
{
    pub fn new(num_features: usize, momentum: T, eps: T, device: Device<B>) -> Self {
        let gamma = Tensor::ones(vec![num_features], device).requires_grad();
        let beta = Tensor::zeros(vec![num_features], device).requires_grad();
        let running_mean = Tensor::zeros(vec![num_features], device);
        let running_var = Tensor::zeros(vec![num_features], device);
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            eps,
        }
    }

    pub fn gamma(&self) -> &Tensor<B, T> {
        &self.gamma
    }

    pub fn beta(&self) -> &Tensor<B, T> {
        &self.beta
    }
}

impl<B: Backend, T: Float> BatchNorm1d<B, T> {
    pub fn forward(&mut self, x: &Tensor<B, T>, is_train: bool) -> Result<Tensor<B, T>> {
        let y = if is_train {
            let (y, mean, var) = batch_norm(x, &self.gamma, &self.beta, 0, self.eps)?;
            let momentum = Tensor::from_scalar(self.momentum, x.device());
            let one = Tensor::from_scalar(T::one(), x.device());
            let running_mean =
                ((&momentum * &self.running_mean)? + ((&one - &momentum)? * mean.detach())?)?;
            let running_var =
                ((&momentum * &self.running_var)? + ((&one - &momentum)? * var.detach())?)?;
            self.running_mean.copy(&running_mean).unwrap();
            self.running_var.copy(&running_var).unwrap();
            y
        } else {
            batch_norm_predict(
                x,
                &self.running_mean,
                &self.running_var,
                &self.gamma,
                &self.beta,
                self.eps,
            )?
        };
        Ok(y)
    }
}

impl<B: Backend, T: Float> Layer<B, T> for BatchNorm1d<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.gamma.clone());
        map.insert("bias".to_string(), self.beta.clone());
        map.insert("running_mean".to_string(), self.running_mean.clone());
        map.insert("running_var".to_string(), self.running_var.clone());
        map
    }

    fn trainable_parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.gamma.clone());
        map.insert("bias".to_string(), self.beta.clone());
        map
    }
}

pub struct BatchNorm2d<B: Backend, T: Float> {
    running_mean: Tensor<B, T>,
    running_var: Tensor<B, T>,
    gamma: Tensor<B, T>,
    beta: Tensor<B, T>,
    momentum: T,
    eps: T,
}

impl<B, T> BatchNorm2d<B, T>
where
    B: Backend,
    T: Float,
{
    pub fn new(num_features: usize, momentum: T, eps: T, device: Device<B>) -> Self {
        let gamma = Tensor::ones(vec![num_features], device).requires_grad();
        let beta = Tensor::zeros(vec![num_features], device).requires_grad();
        let running_mean = Tensor::zeros(vec![1, num_features, 1, 1], device);
        let running_var = Tensor::zeros(vec![1, num_features, 1, 1], device);
        Self {
            gamma,
            beta,
            running_mean,
            running_var,
            momentum,
            eps,
        }
    }

    pub fn gamma(&self) -> &Tensor<B, T> {
        &self.gamma
    }

    pub fn beta(&self) -> &Tensor<B, T> {
        &self.beta
    }
}

impl<B: Backend, T: Float> BatchNorm2d<B, T> {
    pub fn forward(&mut self, x: &Tensor<B, T>, is_train: bool) -> Result<Tensor<B, T>> {
        let y = if is_train {
            let (y, mean, var) = batch_norm2d(x, &self.gamma, &self.beta, self.eps)?;
            let momentum = Tensor::from_scalar(self.momentum, x.device());
            let one = Tensor::from_scalar(T::one(), x.device());
            let running_mean =
                ((&momentum * &self.running_mean)? + ((&one - &momentum)? * mean.detach())?)?;
            let running_var =
                ((&momentum * &self.running_var)? + ((&one - &momentum)? * var.detach())?)?;
            self.running_mean.copy(&running_mean).unwrap();
            self.running_var.copy(&running_var).unwrap();
            y
        } else {
            batch_norm_predict2d(
                x,
                &self.running_mean,
                &self.running_var,
                &self.gamma,
                &self.beta,
                self.eps,
            )?
        };
        Ok(y)
    }
}

impl<B: Backend, T: Float> Layer<B, T> for BatchNorm2d<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.gamma.clone());
        map.insert("bias".to_string(), self.beta.clone());
        map.insert("running_mean".to_string(), self.running_mean.clone());
        map.insert("running_var".to_string(), self.running_var.clone());
        map
    }

    fn trainable_parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.gamma.clone());
        map.insert("bias".to_string(), self.beta.clone());
        map
    }
}

pub fn batch_norm<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    beta: &Tensor<B, T>,
    axis: usize,
    eps: T,
) -> Result<(Tensor<B, T>, Tensor<B, T>, Tensor<B, T>)> {
    let mean = x.mean_axis(axis, false)?;
    let xc = (x - &mean)?;
    let var = xc.pow_scalar(T::from_f64(2.0))?.mean_axis(axis, false)?;
    let std = (&var + eps)?.sqrt();
    let xn = (xc / std)?;
    let y = (gamma * xn + beta)?;
    Ok((y, mean, var))
}

pub fn batch_norm2d<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    beta: &Tensor<B, T>,
    eps: T,
) -> Result<(Tensor<B, T>, Tensor<B, T>, Tensor<B, T>)> {
    let gamma = gamma.reshape(vec![1, gamma.shape()[0], 1, 1])?;
    let beta = beta.reshape(vec![1, beta.shape()[0], 1, 1])?;

    let mean = x.mean_axes(&vec![0, 2, 3], true)?;
    let xc = (x - &mean)?;
    let var = xc
        .pow_scalar(T::from_f64(2.0))?
        .mean_axes(&vec![0, 2, 3], true)?;
    let std = (&var + eps)?.sqrt();
    let xn = (xc / std)?;
    let y = (gamma * xn + beta)?;
    Ok((y, mean, var))
}

pub fn batch_norm_predict<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    running_mean: &Tensor<B, T>,
    running_var: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    beta: &Tensor<B, T>,
    eps: T,
) -> Result<Tensor<B, T>> {
    let xc = (x - running_mean)?;
    let xn = (xc / (running_var + eps)?.sqrt())?;
    gamma * xn + beta
}

pub fn batch_norm_predict2d<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    running_mean: &Tensor<B, T>,
    running_var: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    beta: &Tensor<B, T>,
    eps: T,
) -> Result<Tensor<B, T>> {
    let gamma = gamma.reshape(vec![1, gamma.shape()[0], 1, 1])?;
    let beta = beta.reshape(vec![1, beta.shape()[0], 1, 1])?;

    let xc = (x - running_mean)?;
    let xn = (xc / (running_var + eps)?.sqrt())?;
    gamma * xn + beta
}

pub struct LayerNorm<B: Backend, T: Float> {
    pub gamma: Tensor<B, T>,
    pub beta: Option<Tensor<B, T>>,
    normalize_shape: Vec<usize>,
    eps: T,
}

impl<B: Backend, T: Float> LayerNorm<B, T> {
    pub fn new(normalize_shape: Vec<usize>, eps: T, use_bias: bool, device: Device<B>) -> Self {
        let gamma = Tensor::ones(normalize_shape.clone(), device).requires_grad();
        let beta = if use_bias {
            let beta = Tensor::zeros(normalize_shape.clone(), device).requires_grad();
            Some(beta)
        } else {
            None
        };
        Self {
            gamma,
            beta,
            normalize_shape,
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor<B, T>) -> Result<Tensor<B, T>> {
        let y = if let Some(beta) = self.beta.as_ref() {
            layer_norm(x, &self.gamma, Some(&beta), &self.normalize_shape, self.eps)?
        } else {
            layer_norm(&x, &self.gamma, None, &self.normalize_shape, self.eps)?
        };
        Ok(y)
    }
}

impl<B: Backend, T: Float> Layer<B, T> for LayerNorm<B, T> {
    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        let mut map = HashMap::new();
        map.insert("weight".to_string(), self.gamma.clone());
        if let Some(beta) = self.beta.as_ref() {
            map.insert("bias".to_string(), beta.clone());
        };
        map
    }
}

pub fn layer_norm<B: Backend, T: Float>(
    x: &Tensor<B, T>,
    gamma: &Tensor<B, T>,
    beta: Option<&Tensor<B, T>>,
    normalize_shape: &[usize],
    eps: T,
) -> Result<Tensor<B, T>> {
    let mut axes = Vec::new();
    for i in 0..normalize_shape.len() {
        let axis = i + (x.ndim() - normalize_shape.len());
        if x.shape()[axis] != normalize_shape[i] {
            panic!(
                "x.shape = {:?}, normalize_shape.shape = {:?}",
                x.shape(),
                normalize_shape
            );
        }
        axes.push(axis);
    }

    let mean = x.mean_axes(&axes, true)?;
    let xc = (x - &mean)?;
    let var = xc.pow_scalar(T::from_f64(2.0))?.mean_axes(&axes, true)?;
    let std = (&var + eps)?.sqrt();
    let xn = (xc / std)?;
    let mut y = (gamma * xn)?;
    if let Some(beta) = beta {
        y = (y + beta)?;
    }
    Ok(y)
}
