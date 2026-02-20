use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::tensor::Tensor;

//
// ------ Addition ------
//
impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b
        }
    }
}

// Ref + Ref
impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, other: &Tensor) -> Self::Output {
        let mut op_clone = self.clone();
        op_clone += other;
        op_clone
    }
}

// Owned + Ref
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(mut self, other: &Tensor) -> Self::Output {
        self += other;
        self
    }
}

// Ref + Owned
impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, mut other: Tensor) -> Self::Output {
        other += self;
        other
    }
}

// Owned + Owned
impl Add<Tensor> for Tensor {
    type Output = Self;
    fn add(self, other: Tensor) -> Self::Output {
        self + &other
    }
}

//
// ------ Subtraction ------
//
impl SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a -= b
        }
    }
}

// Ref - Ref
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, other: &Tensor) -> Self::Output {
        let mut op_clone = self.clone();
        op_clone -= other;
        op_clone
    }
}

// Owned - Ref
impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(mut self, other: &Tensor) -> Self::Output {
        self -= other;
        self
    }
}

// Ref - Owned
impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, mut other: Tensor) -> Self::Output {
        other -= self;
        other
    }
}

// Owned - Owned
impl Sub<Tensor> for Tensor {
    type Output = Self;
    fn sub(self, other: Tensor) -> Self::Output {
        self - &other
    }
}


//
// ------ Multiplication ------
//
impl MulAssign<&Tensor> for Tensor {
    fn mul_assign(&mut self, rhs: &Tensor) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a *= b
        }
    }
}

// Ref * Ref
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, other: &Tensor) -> Self::Output {
        let mut op_clone = self.clone();
        op_clone *= other;
        op_clone
    }
}

// Owned * Ref
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(mut self, other: &Tensor) -> Self::Output {
        self *= other;
        self
    }
}

// Ref * Owned
impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, mut other: Tensor) -> Self::Output {
        other *= self;
        other
    }
}

// Owned * Owned
impl Mul<Tensor> for Tensor {
    type Output = Self;
    fn mul(self, other: Tensor) -> Self::Output {
        self * &other
    }
}


//
// ------ Division ------
//
impl DivAssign<&Tensor> for Tensor {
    fn div_assign(&mut self, rhs: &Tensor) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a /= b
        }
    }
}

// Ref / Ref
impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, other: &Tensor) -> Self::Output {
        let mut op_clone = self.clone();
        op_clone /= other;
        op_clone
    }
}

// Owned / Ref
impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(mut self, other: &Tensor) -> Self::Output {
        self /= other;
        self
    }
}

// Ref / Owned
impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, mut other: Tensor) -> Self::Output {
        other /= self;
        other
    }
}

// Owned / Owned
impl Div<Tensor> for Tensor {
    type Output = Self;
    fn div(self, other: Tensor) -> Self::Output {
        self / &other
    }
}

//
// ---- Negation ----
//
impl Neg for Tensor {
    type Output = Tensor;

    fn neg(mut self) -> Self::Output {
        for x in self.data.iter_mut() {
            *x = -(*x);
        }
        self
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_test() {
        let t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let t2 = Tensor::new(vec![-1.,-2.,-3.,-4.,-5.,-6.], vec![2,3]);
        let c = t1 + t2;

        assert_eq!(c.data, vec![0f32; 6]);
    }

    #[test]
    fn sub_test() {
        let t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let t2 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let c = &t1 - &t2;

        assert_eq!(c.data, vec![0f32; 6]);
    }

}
