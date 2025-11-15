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

pub struct Adam<B: Backend, T: Float> {
    alpha: T,
    beta1: T,
    beta2: T,
    eps: T,
    t: T,
    lr: T,
    m: HashMap<String, Tensor<B, T>>,
    v: HashMap<String, Tensor<B, T>>,
}

impl<B: Backend, T: Float> Adam<B, T> {
    pub fn new(alpha: T, beta1: T, beta2: T, eps: T) -> Self {
        Self {
            alpha,
            beta1,
            beta2,
            eps,
            t: T::zero(),
            lr: T::zero(),
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(
            T::from_f64(1e-3),
            T::from_f64(0.9),
            T::from_f64(0.999),
            T::from_f64(1e-8),
        )
    }
}

impl<B: Backend, T: Float> Optimizer<B, T> for Adam<B, T> {
    fn prepare_update_parameters(&mut self) {
        self.t += T::from_f64(1.0);
        // alpha (lr) はそのまま。バイアス補正は後で step_size に入れる
    }

    fn update_parameter(
        &mut self,
        name: &str,
        parameter: &mut Tensor<B, T>,
        grad: &Tensor<B, T>,
    ) -> Result<()> {
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
        let new_m = ((m.clone() * self.beta1)? + (grad * (T::from_f64(1.0) - self.beta1))?)?;

        // 2次モーメント更新: v = beta2 * v + (1 - beta2) * grad^2
        // let grad_sq = &grad * &grad;
        let grad_sq = grad.pow_scalar(T::from_f64(2.0))?;
        let new_v = ((v.clone() * self.beta2)? + (&grad_sq * (T::from_f64(1.0) - self.beta2))?)?;

        // バイアス補正項
        let bias_correction1 = T::from_f64(1.0) - self.beta1.powf(self.t);
        let bias_correction2 = T::from_f64(1.0) - self.beta2.powf(self.t);

        // ステップサイズ
        let step_size = self.alpha / bias_correction1;

        let denom = ((&new_v / bias_correction2)?.sqrt() + self.eps)?;

        // 更新量
        let update = ((&new_m * step_size)? / denom)?;

        // パラメータの更新: p = p - update
        parameter.sub_assign(&update)?;

        // モーメントの保存
        self.m.insert(name.to_string(), new_m);
        self.v.insert(name.to_string(), new_v);

        Ok(())
    }

    fn parameters_map(&self) -> HashMap<String, Tensor<B, T>> {
        todo!()
    }
}
