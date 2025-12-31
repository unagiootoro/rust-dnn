use std::{
    f32, fmt::Debug, iter::Sum, ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign}, u32
};

use rust_dnn_wgpu::wgpu_dtype::WgpuDType;

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
    fn as_u32(self) -> u32;
    fn as_f32(self) -> f32;
    fn as_f64(self) -> f64;
    fn from_isize(value: isize) -> Self;
    fn from_usize(value: usize) -> Self;
    fn from_f64(n: f64) -> Self;
    fn from_f32(n: f32) -> Self;
    fn zero() -> Self;
    fn one() -> Self;
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn dtype() -> DType;
}

impl Num for u32 {
    fn as_usize(self) -> usize {
        self as usize
    }

    fn as_u32(self) -> Self {
        self
    }

    fn as_f32(self) -> f32 {
        self as f32
    }

    fn as_f64(self) -> f64 {
        self as f64
    }

    fn from_isize(value: isize) -> Self {
        value as Self
    }

    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f64(n: f64) -> Self {
        n as Self
    }

    fn from_f32(n: f32) -> Self {
        n as Self
    }

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }

    fn min_value() -> Self {
        u32::MIN
    }

    fn max_value() -> Self {
        u32::MAX
    }

    fn dtype() -> DType {
        DType::U32
    }
}

impl Num for f32 {
    fn as_usize(self) -> usize {
        self as usize
    }

    fn as_u32(self) -> u32 {
        self as u32
    }

    fn as_f32(self) -> f32 {
        self
    }

    fn as_f64(self) -> f64 {
        self as f64
    }

    fn from_isize(value: isize) -> Self {
        value as Self
    }

    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f64(n: f64) -> Self {
        n as Self
    }

    fn from_f32(n: f32) -> Self {
        n as Self
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn min_value() -> Self {
        f32::MIN
    }

    fn max_value() -> Self {
        f32::MAX
    }

    fn dtype() -> DType {
        DType::F32
    }
}

impl Num for f64 {
    fn as_usize(self) -> usize {
        self as usize
    }

    fn as_u32(self) -> u32 {
        self as u32
    }

    fn as_f32(self) -> f32 {
        self as f32
    }

    fn as_f64(self) -> f64 {
        self
    }

    fn from_isize(value: isize) -> Self {
        value as Self
    }

    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f64(n: f64) -> Self {
        n as Self
    }

    fn from_f32(n: f32) -> Self {
        n as Self
    }

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn min_value() -> Self {
        f64::MIN
    }

    fn max_value() -> Self {
        f64::MAX
    }

    fn dtype() -> DType {
        DType::F64
    }
}
