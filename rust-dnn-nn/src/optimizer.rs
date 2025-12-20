use std::collections::HashMap;

use rust_dnn_core::{
    backend::Backend, error::Result, float::Float, gradients::Gradients, tensor::Tensor,
};

pub trait Optimizer<B: Backend, T: Float> {
    fn update_parameters(
        &mut self,
        parameters: &mut HashMap<String, Tensor<B, T>>,
        grads: &Gradients<B, T>,
    ) {
        self.prepare_update_parameters();
        for (name, parameter) in parameters {
            if let Some(grad) = grads.get(parameter) {
                self.update_parameter(name, parameter, &grad.detach());
            } else {
                panic!("{}: grad is None", name);
            }
        }
    }

    fn prepare_update_parameters(&mut self) {}

    fn update_parameter(&mut self, name: &str, parameter: &mut Tensor<B, T>, grad: &Tensor<B, T>);

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>>;

    fn load_parameters_map(&mut self, map: HashMap<String, Tensor<B, T>>) -> Result<()> {
        for (name, parameter) in self.parameters_map() {
            parameter.copy(&map[&name]);
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
    fn update_parameter(&mut self, _name: &str, parameter: &mut Tensor<B, T>, grad: &Tensor<B, T>) {
        parameter.sub_assign(&(self.lr * grad));
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        HashMap::new()
    }
}

pub struct Adam<B: Backend, T: Float> {
    alpha: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: f64,
    lr: f64,
    m: HashMap<String, Tensor<B, T>>,
    v: HashMap<String, Tensor<B, T>>,
}

impl<B: Backend, T: Float> Adam<B, T> {
    pub fn new(alpha: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            eps,
            t: 0.0,
            lr: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(1e-3, 0.9, 0.999, 1e-8)
    }
}

impl<B: Backend, T: Float> Optimizer<B, T> for Adam<B, T> {
    fn prepare_update_parameters(&mut self) {
        self.t += 1.0;
        // alpha (lr) はそのまま。バイアス補正は後で step_size に入れる
    }

    fn update_parameter(&mut self, name: &str, parameter: &mut Tensor<B, T>, grad: &Tensor<B, T>) {
        // m（exp_avg）の取得または初期化
        let m = self
            .m
            .entry(name.to_string())
            .or_insert_with(|| Tensor::zeros(parameter.shape().to_vec(), parameter.device()));

        // v（exp_avg_sq）の取得または初期化
        let v = self
            .v
            .entry(name.to_string())
            .or_insert_with(|| Tensor::zeros(parameter.shape().to_vec(), parameter.device()));

        // 1次モーメント更新: m = beta1 * m + (1 - beta1) * grad
        let new_m = (m.clone() * self.beta1) + (grad * (1.0 - self.beta1));

        // 2次モーメント更新: v = beta2 * v + (1 - beta2) * grad^2
        // let grad_sq = &grad * &grad;
        let grad_sq = grad.pow_scalar(2.0);
        let new_v = (v.clone() * self.beta2) + (&grad_sq * (1.0 - self.beta2));

        // バイアス補正項
        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);

        // ステップサイズ
        let step_size = self.alpha / bias_correction1;

        let denom = (&new_v / bias_correction2).sqrt() + self.eps;

        // 更新量
        let update = (&new_m * step_size) / denom;

        // パラメータの更新: p = p - update
        parameter.sub_assign(&update);

        // モーメントの保存
        self.m.insert(name.to_string(), new_m);
        self.v.insert(name.to_string(), new_v);
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        todo!()
    }
}

pub struct AdamW<B: Backend, T: Float> {
    alpha: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    t: f64,
    lr: f64,
    m: HashMap<String, Tensor<B, T>>,
    v: HashMap<String, Tensor<B, T>>,
}

impl<B: Backend, T: Float> AdamW<B, T> {
    pub fn new(alpha: f64, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0.0,
            lr: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(1e-3, 0.9, 0.999, 1e-8, 1e-2)
    }
}

impl<B: Backend, T: Float> Optimizer<B, T> for AdamW<B, T> {
    fn prepare_update_parameters(&mut self) {
        self.t += 1.0;
        // alpha (lr) はそのまま。バイアス補正は後で step_size に入れる
    }

    fn update_parameter(&mut self, name: &str, parameter: &mut Tensor<B, T>, grad: &Tensor<B, T>) {
        // m（exp_avg）の取得または初期化
        let m = self
            .m
            .entry(name.to_string())
            .or_insert_with(|| Tensor::zeros(parameter.shape().to_vec(), parameter.device()));

        // v（exp_avg_sq）の取得または初期化
        let v = self
            .v
            .entry(name.to_string())
            .or_insert_with(|| Tensor::zeros(parameter.shape().to_vec(), parameter.device()));

        // 1次モーメント更新: m = beta1 * m + (1 - beta1) * grad
        let new_m = (m.clone() * self.beta1) + (grad * (1.0 - self.beta1));

        // 2次モーメント更新: v = beta2 * v + (1 - beta2) * grad^2
        // let grad_sq = &grad * &grad;
        let grad_sq = grad.pow_scalar(2.0);
        let new_v = (v.clone() * self.beta2) + (&grad_sq * (1.0 - self.beta2));

        // バイアス補正項
        let bias_correction1 = 1.0 - self.beta1.powf(self.t);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t);

        // ステップサイズ
        let step_size = self.alpha / bias_correction1;

        let denom = (&new_v / bias_correction2).sqrt() + self.eps;

        // 更新量
        let update = (&new_m * step_size) / denom;

        // パラメータの更新: p = p - update
        parameter.sub_assign(&update);

        // モーメントの保存
        self.m.insert(name.to_string(), new_m);
        self.v.insert(name.to_string(), new_v);

        if self.weight_decay > 0.0 {
            let parameter2 = parameter.clone() * self.alpha * self.weight_decay;
            parameter.sub_assign(&parameter2);
        }
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        todo!()
    }
}
