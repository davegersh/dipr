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

        // += &f32
        impl $assign_trait<&f32> for Tensor {
            fn $assign_func(&mut self, rhs: &f32) {
                for a in self.data.iter_mut() {
                    *a $assign_op rhs
                }
            }
        }

        // Owned + &f32
        impl $op_trait<&f32> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, rhs: &f32) -> Self::Output {
                self $assign_op rhs;
                self
            }
        }

        // Ref + &f32
        impl $op_trait<&f32> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, rhs: &f32) -> Self::Output {
                self.clone() $op rhs
            }
        }

        // += &f32
        impl $assign_trait<f32> for Tensor {
            fn $assign_func(&mut self, rhs: f32) {
                for a in self.data.iter_mut() {
                    *a $assign_op rhs
                }
            }
        }

        // Owned + &f32
        impl $op_trait<f32> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, rhs: f32) -> Self::Output {
                self $assign_op rhs;
                self
            }
        }

        // Ref + &f32
        impl $op_trait<f32> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, rhs: f32) -> Self::Output {
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

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape && self.stride == other.stride
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.rank, other.rank, "Matrix Multiplication only possible with Tensors of equal rank.");

        if self.rank > 2 {
            todo!("Add support for batched matmul! This only works properly with 2D tensors!");
        }

        let rank = self.rank;

        let m = self.shape[rank-2];
        let n = other.shape[rank-1];
        let k = self.shape[rank-1];

        assert_eq!(k, other.shape[rank-2], "Invalid shapes for matmul! Number of columns in 'a' != number of columns in 'b'!");

        let mut new_shape = self.shape.clone();
        new_shape[rank-1] = n;

        let mut new = Tensor::zeros(&new_shape);

        for i in 0..m {
            for j in 0..n {
                let mut dot = 0.0;

                for k in 0..k {
                    dot += self[&[i, k]] * other[&[k, j]];
                }

                new[&[i, j]] = dot; //dot product of row i in A and column j in B
            }
        }

        new
    }

    pub fn permute(&self, perm: &[usize]) -> Tensor {
        let mut new = self.clone();

        for (i, p) in perm.iter().zip(0..self.rank) {
            new.shape[*i] = self.shape[p];
            new.stride[*i] = self.stride[p]
        }

        new
    }

    pub fn transpose(&self) -> Tensor {
        let mut perm: Vec<usize> = (0..self.rank).collect();

        // swap last two shapes and strides for matrix transposition
        perm.swap(0, 1);

        self.permute(&perm)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_test() {
        let t1 = Tensor::new(vec![4., 2., -3., 1.], vec![2,2]);
        let t2 = Tensor::new(vec![1., 5., 3., 2., 7., -4.,], vec![2,3]);

        let m = Tensor::matmul(&t1, &t2);
        let expected = Tensor::new(vec![8., 34., 4., -1., -8., -13.], vec![2,3]);

        assert_eq!(m, expected);
    }

    #[test]
    fn transpose_test() {
        let t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        assert_eq!(t1.permute(&[1,0]), t1.transpose());
    }

    #[test]
    fn add_test() {
        let mut t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let t2 = Tensor::new(vec![-1.,-2.,-3.,-4.,-5.,-6.], vec![2,3]);

        t1 += &t2;

        assert_eq!(t1, Tensor::zeros(&[2,3]));

        t1 += 1.;

        assert_eq!(t1, Tensor::ones(&[2,3]));
    }

    #[test]
    fn sub_test() {
        let mut t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let t2 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);

        t1 -= &t2;

        assert_eq!(t1, Tensor::zeros(&[2,3]));

        t1 -= 1.;

        assert_eq!(t1, Tensor::fill(&[2,3], -1.));
    }

    #[test]
    fn mul_test() {
        let mut t1 = Tensor::ones(&[2,3]);
        let t2 = Tensor::fill(&[2,3], 2.);

        t1 *= &t2;

        assert_eq!(t1, Tensor::fill(&[2,3], 2.));

        t1 *= 0.;

        assert_eq!(t1, Tensor::zeros(&[2,3]));
    }

    #[test]
    fn div_test() {
        let mut t1 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        let t2 = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);

        t1 /= &t2;

        assert_eq!(t1, Tensor::ones(&[2,3]));

        t1 /= 2.;

        assert_eq!(t1, Tensor::fill(&[2,3], 0.5));
    }




}
