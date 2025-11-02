use crate::{backend::Backend, dim::DynDim, num::Num, tensor::Tensor};

#[derive(Clone)]
pub enum Op<B: Backend, T: Num> {
    Reshape(Tensor<B, DynDim, T>),
    PermutedAxes(Tensor<B, DynDim, T>, Vec<usize>),
    BroadcastTo(Tensor<B, DynDim, T>),
    Contiguous(Tensor<B, DynDim, T>),
    SumAxis(Tensor<B, DynDim, T>, usize, bool),
    Add(Tensor<B, DynDim, T>, Tensor<B, DynDim, T>),
    Sub(Tensor<B, DynDim, T>, Tensor<B, DynDim, T>),
    Mul(Tensor<B, DynDim, T>, Tensor<B, DynDim, T>),
    Div(Tensor<B, DynDim, T>, Tensor<B, DynDim, T>),
    Neg(Tensor<B, DynDim, T>),
    Pow(Tensor<B, DynDim, T>, Tensor<B, DynDim, T>),
    Ln(Tensor<B, DynDim, T>),
}
