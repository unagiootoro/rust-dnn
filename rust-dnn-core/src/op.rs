use crate::{backend::Backend, num::Num, tensor::Tensor};

#[derive(Clone)]
pub enum Op<B: Backend, T: Num> {
    Reshape(Tensor<B, T>),
    PermutedAxes(Tensor<B, T>, Vec<usize>),
    BroadcastTo(Tensor<B, T>),
    GetItem(Tensor<B, T>, Vec<(usize, usize)>),
    Contiguous(Tensor<B, T>),
    Sum(Tensor<B, T>),
    SumAxis(Tensor<B, T>),
    Max(Tensor<B, T>, Tensor<B, T>),
    MaxAxis(Tensor<B, T>, Tensor<B, T>),
    Add(Tensor<B, T>, Tensor<B, T>),
    Sub(Tensor<B, T>, Tensor<B, T>),
    Mul(Tensor<B, T>, Tensor<B, T>),
    Div(Tensor<B, T>, Tensor<B, T>),
    Neg(Tensor<B, T>),
    Matmul(Tensor<B, T>, Tensor<B, T>),
    Pow(Tensor<B, T>, Tensor<B, T>),
    Exp(Tensor<B, T>),
    Ln(Tensor<B, T>),
    Gather(Tensor<B, T>, Tensor<B, u32>, usize),
    IndexSelect(Tensor<B, T>, Tensor<B, u32>, usize),
}
