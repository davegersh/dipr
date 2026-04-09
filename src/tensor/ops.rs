use super::Tensor;

impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        match (self.rank, other.rank) {
            (2, 2) => self.matmul_2d(other),
            (3, 3) => self.matmul_batch(other),
            (3, 2) => {
                let mut new_shape = other.shape.clone();
                new_shape.insert(0, 1);

                let reshaped_other = other.reshape(&new_shape);
                self.matmul_batch(&reshaped_other)
            }
            (2, 3) => {
                let mut new_shape = self.shape.clone();
                new_shape.insert(0, 1);

                let reshaped_self = self.reshape(&new_shape);
                reshaped_self.matmul_batch(other)
            }
            (r1, r2) if r1 > 3 && r2 > 3 => {
                // flatten self and other batch dimensions to a rank 3 tensor
                let flat_self = self.flatten_left(3);
                let flat_other = other.flatten_left(3);

                // perform a matrix multiplication on each flattened batch
                let mut result = flat_self.matmul_batch(&flat_other);

                // take original batch dimensions
                let mut new_shape = self.shape[0..self.rank - 2].to_vec();

                // add new 2d matrix multiplication shapes
                new_shape.push(result.shape[1]);
                new_shape.push(result.shape[2]);

                // reshape matrix multiplication back to original shape
                result.reshape_mut(&new_shape);

                result
            }
            _ => {
                panic!(
                    "Cannot complete matrix multiplication with ranks: {:?} @ {:?}",
                    self.rank, other.rank
                )
            }
        }
    }

    pub fn matmul_2d(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.rank == 2,
            other.rank == 2,
            "2D Matrix Multiplication requires two rank 2 tensors!"
        );

        let rank = self.rank;

        let m = self.shape[rank - 2];
        let n = other.shape[rank - 1];
        let k = self.shape[rank - 1];

        assert_eq!(
            k,
            other.shape[rank - 2],
            "Invalid shapes for matmul! Inner shapes must match: {:?} * {:?}",
            self.shape,
            other.shape
        );

        let mut new_shape = self.shape.clone();
        new_shape[rank - 1] = n;

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

    pub fn matmul_batch(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.rank == 3,
            other.rank == 3,
            "Batch Matrix Multiplication requires two rank 3 tensors!"
        );

        // broadcast only the batch dimension (handles the case if self or other is 1)
        let b_dim = self.broadcast_dim(other, 0);

        assert_ne!(b_dim, None, "Cannot broadcast batch dimension for matmul!");

        let rank = self.rank;

        let b = b_dim.unwrap();
        let m = self.shape[rank - 2];
        let n = other.shape[rank - 1];
        let k = self.shape[rank - 1];

        assert_eq!(
            k,
            other.shape[rank - 2],
            "Invalid shapes for matmul! Inner shapes must match: {:?} @ {:?}",
            self.shape,
            other.shape
        );

        let mut new_shape = self.shape.clone();
        new_shape[rank - 1] = n;

        let mut new = Tensor::zeros(&new_shape);

        for b in 0..b {
            for i in 0..m {
                for j in 0..n {
                    let mut dot = 0.0;
                    for k in 0..k {
                        dot += self[&[b, i, k]] * other[&[b, k, j]];
                    }
                    new[&[b, i, j]] = dot; //dot product of row i in A and column j in B
                }
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
        perm.swap(self.rank - 2, self.rank - 1);

        self.permute(&perm)
    }

    // Other element-wise ops
    pub fn map_mut<F>(&mut self, f: F) -> &mut Self
    where
        F: Fn(f32) -> f32,
    {
        for val in self.data.iter_mut() {
            *val = f(*val);
        }

        self
    }

    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let mut new = self.clone();
        new.map_mut(f);
        new
    }

    pub fn zero(&mut self) {
        self.map_mut(|_| 0.0);
    }

    pub fn ln(&self) -> Tensor {
        self.map(f32::ln)
    }

    pub fn exp(&self) -> Tensor {
        self.map(f32::exp)
    }

    pub fn sqrt(&self) -> Tensor {
        self.map(f32::sqrt)
    }

    pub fn softmax(&self) -> Tensor {
        let max = self.max(self.rank - 1);

        let mut exp = (self - max).exp();

        let exp_sum = exp.sum(self.rank - 1);

        exp /= &(exp_sum + 1e-8);
        exp
    }

    // Reduction Ops
    pub fn reduce<F>(&self, axis: usize, op: F) -> Self
    where
        F: Fn(&Vec<f32>) -> f32,
    {
        let mut new_shape = self.shape.clone();
        new_shape[axis] = 1;

        let pre: usize = self.shape.iter().take(axis).product();
        let slice_size: usize = self.shape[axis];
        let post: usize = self.shape.iter().skip(axis + 1).product();

        let mut output_data = Vec::with_capacity(post);

        for i in 0..pre {
            for j in 0..post {
                // Collect axis slice elements
                let mut slice = Vec::with_capacity(slice_size);

                for k in 0..slice_size {
                    let idx = (i * slice_size * post) + (k * post) + j;
                    slice.push(self.data[idx]);
                }

                // Calculate on slice
                let result = op(&slice);
                output_data.push(result);
            }
        }

        Tensor::new(output_data, new_shape)
    }

    pub fn sum(&self, axis: usize) -> Tensor {
        self.reduce(axis, |slice| slice.iter().sum())
    }

    pub fn sum_all(&self) -> Tensor {
        let sum = self.data.iter().sum();

        let mut new_shape = self.shape.clone();
        new_shape.fill(1);

        Tensor::new(vec![sum], new_shape)
    }

    pub fn max(&self, axis: usize) -> Tensor {
        self.reduce(axis, |slice| {
            slice.iter().copied().max_by(f32::total_cmp).unwrap()
        })
    }

    pub fn min(&self, axis: usize) -> Tensor {
        self.reduce(axis, |slice| {
            slice.iter().copied().min_by(f32::total_cmp).unwrap()
        })
    }

    pub fn mean(&self, axis: usize) -> Tensor {
        self.reduce(axis, |slice| {
            (slice.iter().sum::<f32>()) / self.shape[axis] as f32
        })
    }

    pub fn std(&self, axis: usize) -> Tensor {
        let mean = self.mean(axis);
        (self - mean).map_mut(|x| x * x).mean(axis)
    }

    pub fn normalize(&self, axis: usize) -> Tensor {
        // 1. Calculate Mean (Collapse rows, keep columns)
        let mu = self.mean(axis);

        // 2. Calculate Std Dev (The "spread")
        let sigma = self.std(axis);

        (self - mu) / (sigma + 1e-8)
    }

    pub fn min_max_scale(&self, axis: usize) -> Tensor {
        let min = self.min(axis);
        let max = self.max(axis);

        (self - &min) / (max - &min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let t1 = Tensor::new(vec![4.0, 2.0, -3.0, 1.0], vec![2, 2]);
        let t2 = Tensor::new(vec![1.0, 5.0, 3.0, 2.0, 7.0, -4.0], vec![2, 3]);

        let m = Tensor::matmul_2d(&t1, &t2);
        let expected = Tensor::new(vec![8.0, 34.0, 4.0, -1.0, -8.0, -13.0], vec![2, 3]);

        assert_eq!(m, expected);
    }

    #[test]
    fn test_matmul_3d() {
        let t1 = Tensor::new(
            vec![4.0, 2.0, -3.0, 1.0, 4.0, 2.0, -3.0, 1.0],
            vec![2, 2, 2],
        );
        let t2 = Tensor::new(
            vec![1.0, 5.0, 3.0, 2.0, 7.0, -4.0, 1.0, 5.0, 3.0, 2.0, 7.0, -4.0],
            vec![2, 2, 3],
        );

        let m = Tensor::matmul_batch(&t1, &t2);
        let expected = Tensor::new(
            vec![
                8.0, 34.0, 4.0, -1.0, -8.0, -13.0, 8.0, 34.0, 4.0, -1.0, -8.0, -13.0,
            ],
            vec![2, 2, 3],
        );

        assert_eq!(m, expected);
    }

    #[test]
    fn test_transpose() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t1.permute(&[1, 0]), t1.transpose());

        // 3D "transpose" test (just swaps last two shapes)
        let t1 = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        );
        assert_eq!(t1.permute(&[0, 2, 1]), t1.transpose());
    }

    #[test]
    fn test_sum() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        assert_eq!(t1.sum(0), Tensor::new(vec![9.0, 12.0], vec![1, 2]));
        assert_eq!(t1.sum(1), Tensor::new(vec![3.0, 7.0, 11.0,], vec![3, 1]));
    }

    #[test]
    fn test_max() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        assert_eq!(t1.max(0), Tensor::new(vec![5.0, 6.0], vec![1, 2]));
        assert_eq!(t1.max(1), Tensor::new(vec![2.0, 4.0, 6.0,], vec![3, 1]));
    }

    #[test]
    fn test_min() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        assert_eq!(t1.min(0), Tensor::new(vec![1.0, 2.0], vec![1, 2]));
        assert_eq!(t1.min(1), Tensor::new(vec![1.0, 3.0, 5.0,], vec![3, 1]));
    }

    #[test]
    fn test_mean() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        assert_eq!(t1.mean(0), Tensor::new(vec![3.0, 4.0], vec![1, 2]));
        assert_eq!(t1.mean(1), Tensor::new(vec![1.5, 3.5, 5.5,], vec![3, 1]));
    }

    #[test]
    fn test_std() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        assert_eq!(
            t1.std(0),
            Tensor::new(vec![8.0 / 3.0, 8.0 / 3.0], vec![1, 2])
        );
        assert_eq!(t1.std(1), Tensor::new(vec![0.25, 0.25, 0.25], vec![3, 1]));
    }
}
