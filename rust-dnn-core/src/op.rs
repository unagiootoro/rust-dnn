use crate::{num::Num, tensor::Tensor};

#[derive(Clone)]
pub enum Op<T: Num> {
    Reshape(Tensor<T>),
}
