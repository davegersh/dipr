use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use super::tensor::Tensor;

// Handles implementation of any element wise operation (including scalar)
macro_rules! element_op {
    ($assign_trait:ident, $assign_func:ident, $assign_op:tt, $op_trait:ident, $op_func:ident, $op:tt) => {
        // Assignment
        impl $assign_trait<&Tensor> for Tensor {
            fn $assign_func(&mut self, rhs: &Tensor) {
                for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                    *a $assign_op b
                }
            }
        }

        // Ref + Ref
        impl $op_trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, other: &Tensor) -> Self::Output {
                let mut op_clone = self.clone();
                op_clone $assign_op other;
                op_clone
            }
        }

        // Owned + Ref
        impl $op_trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, other: &Tensor) -> Self::Output {
                self $assign_op other;
                self
            }
        }

        // Ref + Owned
        impl $op_trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, mut other: Tensor) -> Self::Output {
                other $assign_op self;
                other
            }
        }

        // Owned + Owned
        impl $op_trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $op_func(self, other: Tensor) -> Self::Output {
                self $op &other
            }
        }

        // += f32
        impl $assign_trait<&f32> for Tensor {
            fn $assign_func(&mut self, rhs: &f32) {
                for a in self.data.iter_mut() {
                    *a $assign_op rhs
                }
            }
        }

        // Owned + f32
        impl $op_trait<&f32> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, rhs: &f32) -> Self::Output {
                self $assign_op rhs;
                self
            }
        }

        // Owned + f32
        impl $op_trait<&f32> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, rhs: &f32) -> Self::Output {
                self.clone() $op rhs
            }
        }
    };
}

//
// ---- Addition, Subtraction, Multiplication, Division -----
//
element_op!(AddAssign, add_assign, +=, Add, add, +);
element_op!(SubAssign, sub_assign, -=, Sub, sub, -);
element_op!(MulAssign, mul_assign, *=, Mul, mul, *);
element_op!(DivAssign, div_assign, /=, Div, div, /);

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
