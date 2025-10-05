use crate::{num::Num, tensor::Tensor};

#[derive(Clone)]
pub enum Op<T: Num> {
    Reshape(Tensor<T>),
    PermutedAxes(Tensor<T>, Vec<usize>),
    BroadcastTo(Tensor<T>),
    Contiguous(Tensor<T>),
    SumAxis(Tensor<T>, usize, bool),
    Add(Tensor<T>, Tensor<T>),
    Sub(Tensor<T>, Tensor<T>),
    Mul(Tensor<T>, Tensor<T>),
    Div(Tensor<T>, Tensor<T>),
    Neg(Tensor<T>),
    PowScalar(Tensor<T>, T),
}
