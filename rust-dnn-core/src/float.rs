use std::ops::Neg;

use crate::num::Num;

pub trait Float: Num + Neg<Output = Self> {
    fn powf(self, rhs: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
}

impl Float for f32 {
    fn powf(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tanh(self) -> Self {
        self.tanh()
    }
}

impl Float for f64 {
    fn powf(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tanh(self) -> Self {
        self.tanh()
    }
}
