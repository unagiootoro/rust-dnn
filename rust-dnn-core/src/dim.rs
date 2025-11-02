#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimKind {
    ConstDim,
    DynDim,
}

pub trait Dim: Clone {
    fn kind() -> DimKind;
    fn dim() -> usize;
}

#[derive(Clone)]
pub struct ConstDim<const N: usize>;

impl<const N: usize> Dim for ConstDim<N> {
    fn kind() -> DimKind {
        DimKind::ConstDim
    }

    fn dim() -> usize {
        N
    }
}

#[derive(Clone)]
pub struct DynDim;

impl Dim for DynDim {
    fn kind() -> DimKind {
        DimKind::DynDim
    }

    fn dim() -> usize {
        panic!();
    }
}
