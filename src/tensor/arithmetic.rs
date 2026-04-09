use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Tensor;

impl Tensor {
    // Returns the size of an axis when broadcasted to another tensor at the same axis
    pub fn broadcast_dim(&self, other: &Tensor, axis: usize) -> Option<usize> {
        let dim1 = self.shape[axis];
        let dim2 = other.shape[axis];

        if dim1 == dim2 || dim2 == 1 {
            Some(dim1)
        } else if dim1 == 1 {
            Some(dim2)
        } else {
            None
        }
    }

    /// Returns the shape of a new tensor after ops broadcasting to another tensor (Returns None if not broadcastable)
    pub fn broadcast_shape(&self, other: &Tensor) -> Option<Vec<usize>> {
        if self.shape == other.shape {
            return Some(self.shape.clone());
        }

        let max_rank = self.rank.max(other.rank);
        let mut new_shape = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let self_index = (i as i32) + (self.rank as i32) - (max_rank as i32);
            let other_index = (i as i32) + (other.rank as i32) - (max_rank as i32);

            if self_index < 0 {
                new_shape.push(other.shape[i]);
                continue;
            } else if other_index < 0 {
                new_shape.push(self.shape[i]);
                continue;
            }

            let self_dim = self.shape[self_index as usize];
            let other_dim = other.shape[other_index as usize];

            if self_dim != other_dim {
                if self_dim != 1 && other_dim != 1 {
                    return None;
                }
                new_shape.push(self_dim.max(other_dim));
            } else {
                new_shape.push(self_dim);
            }
        }

        Some(new_shape)
    }
}

// Macro for overloading operators for element-wise ops (including scalar)
macro_rules! arithmetic_op {
    ($assign_trait:ident, $assign_func:ident, $assign_op:tt, $op_trait:ident, $op_func:ident, $op:tt) => {
        // Owned += Ref
        impl $assign_trait<&Tensor> for Tensor {
            fn $assign_func(&mut self, rhs: &Tensor) {

                // Easy element-wise op for tensors of exactly the same shape
                if self.shape == rhs.shape {
                    for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
                        *a $assign_op b
                    }
                }

                // Element-wise operation with broadcasting
                else if let Some(new_shape) = self.broadcast_shape(rhs) {
                    let mut op_tensor = Tensor::zeros(&new_shape);

                    let mn = new_shape.iter().product();

                    for i in 0..mn {
                        let coords = op_tensor.flat_index_to_coords(i);

                        let mut self_index = 0;
                        let mut rhs_index = 0;

                        for k in 0..coords.len() {
                            if self.shape[k] > 1 {
                                self_index += coords[k] * self.stride[k];
                            }

                            if rhs.shape[k] > 1 {
                                rhs_index += coords[k] * rhs.stride[k];
                            }
                        }

                        op_tensor[&coords] = self.data[self_index] $op rhs.data[rhs_index];
                    }

                    *self = op_tensor;
                }

                // Can't broadcast! Panic!
                else {
                    panic!(
                        "Cannot {} element-wise tensors with shapes: {:?} and {:?}! They cannot be broadcasted!",
                        stringify!($op_func), self.shape, rhs.shape
                    )
                }
            }
        }

        // Ref + Ref
        impl $op_trait<&Tensor> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, other: &Tensor) -> Self::Output {
                let self_clone = self.clone();
                self_clone $op other // Owned + Ref
            }
        }

        // Owned + Ref
        impl $op_trait<&Tensor> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, other: &Tensor) -> Self::Output {
                self $assign_op other; // Owned += Ref
                self
            }
        }

        // Ref + Owned
        impl $op_trait<Tensor> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, other: Tensor) -> Self::Output {
                self $op &other // Ref + Ref
            }
        }

        // Owned + Owned
        impl $op_trait<Tensor> for Tensor {
            type Output = Tensor;
            fn $op_func(self, other: Tensor) -> Self::Output {
                self $op &other // Owned + Ref
            }
        }

        // Owned += &f32
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

        // Owned += f32
        impl $assign_trait<f32> for Tensor {
            fn $assign_func(&mut self, rhs: f32) {
                for a in self.data.iter_mut() {
                    *a $assign_op rhs
                }
            }
        }

        // Owned + f32
        impl $op_trait<f32> for Tensor {
            type Output = Tensor;
            fn $op_func(mut self, rhs: f32) -> Self::Output {
                self $assign_op rhs;
                self
            }
        }

        // Ref + f32
        impl $op_trait<f32> for &Tensor {
            type Output = Tensor;
            fn $op_func(self, rhs: f32) -> Self::Output {
                self.clone() $op rhs
            }
        }

        // f32 + Ref
        impl $op_trait<&Tensor> for f32 {
            type Output = Tensor;
            fn $op_func(self, rhs: &Tensor) -> Self::Output {
                let mut new = rhs.clone();
                for a in new.data.iter_mut() {
                    *a = self $op *a;
                }
                new
            }
        }
    };
}

// Addition, Subtraction, Multiplication, Division Implementations
arithmetic_op!(AddAssign, add_assign, +=, Add, add, +);
arithmetic_op!(SubAssign, sub_assign, -=, Sub, sub, -);
arithmetic_op!(MulAssign, mul_assign, *=, Mul, mul, *);
arithmetic_op!(DivAssign, div_assign, /=, Div, div, /);

// Negation
impl Neg for Tensor {
    type Output = Tensor;

    fn neg(mut self) -> Self::Output {
        self.map_mut(|x| -x);
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
    fn test_broadcast_shape_same_size() {
        let t1 = Tensor::zeros(&[3, 2]);
        let t2 = Tensor::zeros(&[3, 1]);
        let broadcast_shape = t1.broadcast_shape(&t2);

        assert_eq!(broadcast_shape, Some(vec![3, 2]));

        let t1 = Tensor::zeros(&[1, 1]);
        let t2 = Tensor::zeros(&[3, 3]);
        let broadcast_shape = t1.broadcast_shape(&t2);

        assert_eq!(broadcast_shape, Some(vec![3, 3]));
    }

    #[test]
    fn test_broadcast_shape_diff_size() {
        let t1 = Tensor::zeros(&[5, 1, 4]);
        let t2 = Tensor::zeros(&[3, 1]);
        let broadcast_shape = t1.broadcast_shape(&t2);

        assert_eq!(broadcast_shape, Some(vec![5, 3, 4]));
    }

    #[test]
    fn test_broadcast_shape_none() {
        let t1 = Tensor::zeros(&[5, 2, 4]);
        let t2 = Tensor::zeros(&[3, 1]);
        let broadcast_shape = t1.broadcast_shape(&t2);

        assert_eq!(broadcast_shape, None);
    }

    // Macro for overloading operators for element-wise ops (including scalar)
    macro_rules! arithmetic_op_test {
        ($op:tt, $tensor_test_name:ident, $scalar_test_name:ident, $broadcast_test_name:ident) => {
            #[test]
            fn $tensor_test_name() {
                let t1 = Tensor::arange(&[2, 3], 1..7);
                let t2 = Tensor::arange(&[2, 3], 7..13);

                let expected_data = t1.data.iter().zip(&t2.data).map(|(x, y)| x $op y).collect();
                let expected = Tensor::new(expected_data, vec![2, 3]);

                assert_eq!(&t1 $op &t2, expected); // Ref + Ref
                assert_eq!(t1.clone() $op &t2, expected); // Owned + Ref
                assert_eq!(&t1 $op t2.clone(), expected); // Ref + Owned
                assert_eq!(t1 $op t2, expected); // Owned + Owned
            }

            #[test]
            fn $scalar_test_name() {
                let t1 = Tensor::arange(&[2, 3], 1..7);

                let scalar = 42.0;

                // Tensor + f32
                let expected_data: Vec<f32> = t1.data.iter().map(|x| x $op scalar).collect();
                let expected = Tensor::new(expected_data, vec![2, 3]);

                assert_eq!(&t1 $op scalar, expected);
                assert_eq!(t1.clone() $op scalar, expected);

                // f32 + Tensor
                let expected_data: Vec<f32> = t1.data.iter().map(|x| scalar $op x).collect();
                let expected = Tensor::new(expected_data, vec![2, 3]);

                assert_eq!(scalar $op &t1, expected);
            }

            #[test]
            fn $broadcast_test_name() {
                let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
                let t2 = Tensor::new(vec![-1.0, -2.0, -3.0], vec![3, 1]);

                let t2_broadcast = Tensor::new(vec![-1.0, -1.0, -2.0, -2.0, -3.0, -3.0], vec![3, 2]);

                assert_eq!(&t1 $op t2, &t1 $op t2_broadcast);
            }
        };
    }

    arithmetic_op_test!(+, test_tensor_add, test_scalar_add, test_broadcast_add);
    arithmetic_op_test!(-, test_tensor_sub, test_scalar_sub, test_broadcast_sub);
    arithmetic_op_test!(*, test_tensor_mul, test_scalar_mul, test_broadcast_mul);
    arithmetic_op_test!(/, test_tensor_div, test_scalar_div, test_broadcast_div);
}
