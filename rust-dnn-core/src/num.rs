use std::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use crate::dtype::DType;

pub trait Num:
    Default
    + Debug
    + Copy
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Sum
    + PartialOrd
    + Sized
    + Send
    + Sync
    + 'static
{
    fn as_usize(self) -> usize;
    fn as_f32(self) -> f32;
    fn zero() -> Self;
    fn one() -> Self;
    fn dtype() -> DType;
}

impl Num for u32 {
    fn as_usize(self) -> usize {
        self as usize
    }

    fn as_f32(self) -> f32 {
        self as f32
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn dtype() -> DType {
        DType::U32
    }
}

impl Num for f32 {
    fn as_usize(self) -> usize {
        self as usize
    }

    fn as_f32(self) -> f32 {
        self
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn dtype() -> DType {
        DType::F32
    }
}
