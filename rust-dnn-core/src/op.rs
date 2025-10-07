use crate::{backend::Backend, num::Num, tensor::Tensor};

#[derive(Clone)]
pub enum Op<B: Backend, T: Num> {
    Reshape(Tensor<B, T>),
    PermutedAxes(Tensor<B, T>, Vec<usize>),
    BroadcastTo(Tensor<B, T>),
    Contiguous(Tensor<B, T>),
    SumAxis(Tensor<B, T>, usize, bool),
    Add(Tensor<B, T>, Tensor<B, T>),
    Sub(Tensor<B, T>, Tensor<B, T>),
    Mul(Tensor<B, T>, Tensor<B, T>),
    Div(Tensor<B, T>, Tensor<B, T>),
    Neg(Tensor<B, T>),
    PowScalar(Tensor<B, T>, T),
}
