pub mod arithmetic;
pub mod ops;

use super::rand::XorShift;
use std::ops::{Index, IndexMut, Range};

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>, //uses a 1D vector to handle any dimension
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub rank: usize,
}

impl Tensor {
    fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];

        strides[shape.len() - 1] = 1;

        for i in (0..shape.len() - 1).rev() {
            if shape[i] > 1 {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        strides
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Length of data array for a new tensor should match the total number of elements for the shape!"
        );

        let stride = Self::shape_to_stride(&shape);
        let rank = shape.len();

        Self {
            data,
            shape,
            stride,
            rank,
        }
    }

    /// Returns a new tensor with data consiting entirely of the given fill value
    pub fn fill(shape: &[usize], fill_value: f32) -> Self {
        let total_elements = shape.iter().product();
        let data = vec![fill_value; total_elements];

        Self::new(data, shape.to_vec())
    }

    /// Returns a new tensor with data consiting entirely 0.0
    pub fn zeros(shape: &[usize]) -> Self {
        Self::fill(shape, 0.)
    }

    /// Returns a new tensor with data consiting entirely 1.0
    pub fn ones(shape: &[usize]) -> Self {
        Self::fill(shape, 1.)
    }

    /// Returns a new tensor filled with data defined by collecting the given range
    pub fn arange(shape: &[usize], range: Range<i32>) -> Self {
        let data: Vec<f32> = range.map(|x| x as f32).collect();
        Self::new(data, shape.to_vec())
    }

    /// Returns a tensor with data made of increasing integer values starting from 1
    pub fn iota(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();
        Self::arange(shape, 1..(total_elements as i32) + 1)
    }

    pub fn rand(shape: &[usize], seed: u32) -> Self {
        let total_elements = shape.iter().product();
        let data = XorShift::new(seed, true)
            .take(total_elements)
            .map(|x| x as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }

    pub fn coords_to_index(&self, coords: &[usize]) -> usize {
        let mut index = 0;

        assert_eq!(
            self.rank,
            coords.len(),
            "Cannot index with coordinates larger than the rank of the tensor!"
        );

        for i in 0..coords.len() {
            index += coords[i] * self.stride[i];
        }
        index
    }

    pub fn flat_index_to_coords(&self, mut index: usize) -> Vec<usize> {
        let mut coords = vec![0; self.rank];

        for i in (0..self.rank).rev() {
            coords[i] = index % self.shape[i];
            index /= self.shape[i];
        }

        coords
    }

    // Reshaping
    pub fn reshape_mut(&mut self, new_shape: &[usize]) {
        let new_total_elements: usize = new_shape.iter().product();

        assert_eq!(
            new_total_elements,
            self.shape.iter().product(),
            "New shape for reshape must have the same number of total elements!"
        );

        self.shape = new_shape.to_vec();
        self.stride = Self::shape_to_stride(&new_shape);
        self.rank = new_shape.len();
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let mut new = self.clone();
        new.reshape_mut(new_shape);

        new
    }

    /// Returns a new tensor with shape flattened to rank 1.
    /// Ex. The shape [B, M, N] becomes [B * M * N]
    pub fn flatten(&self) -> Tensor {
        self.reshape(&[self.shape.iter().product()])
    }

    /// Returns a new tensor based on this tensors shape to a target rank.
    /// The shape is calculated by multiplying the shape array from the left until the target rank is met.
    /// Ex. For a target rank of 3, the shape [B, H, M, N] becomes [B*H, M, N].
    pub fn flatten_left(&self, target_rank: usize) -> Tensor {
        assert!(
            target_rank < self.rank,
            "Target rank for flattening must be smaller than current rank!"
        );

        let rank_diff = self.rank - target_rank;

        let new_shape = &mut self.shape.clone()[rank_diff..self.rank];

        let flattened_elements: usize = self.shape[0..rank_diff].iter().product();
        new_shape[0] *= flattened_elements;

        self.reshape(new_shape)
    }

    /// Returns a new tensor based on this tensors shape to a target rank.
    /// The shape is calculated by multiplying the shape array from the right until the target rank is met.
    /// Ex. For a target rank of 3, the shape [B, H, M, N] becomes [B, H, M*N].
    pub fn flatten_right(&self, target_rank: usize) -> Tensor {
        assert!(
            target_rank < self.rank,
            "Target rank for flattening must be smaller than current rank!"
        );

        let new_shape = &mut self.shape.clone()[0..target_rank];

        let flattened_elements: usize = self.shape[target_rank..self.rank].iter().product();
        new_shape[target_rank - 1] *= flattened_elements;

        self.reshape(new_shape)
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

impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, coords: &[usize]) -> &Self::Output {
        let index = self.coords_to_index(coords);
        &self.data[index]
    }
}

impl IndexMut<&[usize]> for Tensor {
    fn index_mut(&mut self, coords: &[usize]) -> &mut Self::Output {
        let index = self.coords_to_index(coords);
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill() {
        let t = Tensor::fill(&[2, 3], 42.);
        assert_eq!(t.data, vec![42., 42., 42., 42., 42., 42.]);
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(&[2, 3], 0..6);
        assert_eq!(t.data, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_rand() {
        let t = Tensor::rand(&[2, 3], 42);

        for i in 0..6 {
            for j in 0..6 {
                if i != j {
                    assert_ne!(t.data[i], t.data[j])
                }
            }
        }
    }

    #[test]
    fn test_shape_to_stride() {
        // 2D
        let stride1 = Tensor::shape_to_stride(&[2, 3]);
        assert_eq!(stride1, vec![3, 1]);

        // 3D
        let stride2 = Tensor::shape_to_stride(&[2, 2, 3]);
        assert_eq!(stride2, vec![6, 3, 1]);

        // 3D with flat shape
        let stride3 = Tensor::shape_to_stride(&[1, 3, 4]);
        assert_eq!(stride3, vec![0, 4, 1]);
    }

    #[test]
    fn test_index() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t1[&[0, 1]], 2.0);

        let t2 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert_eq!(t2[&[0, 1]], 2.0);

        let t3 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 3, 2]);
        assert_eq!(t3[&[1, 0, 1]], 2.0);
    }

    #[test]
    #[should_panic]
    fn test_reshape_panic() {
        let t1 = Tensor::zeros(&[1, 2, 3, 4]);

        // Panics due to new shape containing more elements than original
        t1.reshape(&[100]);
    }

    #[test]
    fn test_reshape() {
        let t1 = Tensor::zeros(&[1, 2, 3, 4]);

        let t2 = t1.reshape(&[6, 4]);
        assert_eq!(t2.shape, vec![6, 4]);
        assert_eq!(t2.stride, vec![4, 1]);
    }

    #[test]
    fn test_flatten() {
        let t1 = Tensor::zeros(&[1, 2, 3, 4]);

        let t2 = t1.flatten();
        assert_eq!(t2.shape, vec![24]);
    }

    #[test]
    fn test_flatten_left() {
        let t1 = Tensor::zeros(&[1, 2, 3, 4]);

        let t2 = t1.flatten_left(2);
        assert_eq!(t2.shape, vec![6, 4]);

        let t3 = t1.flatten_left(1);
        assert_eq!(t3.shape, vec![24]);
    }

    #[test]
    fn test_flatten_right() {
        let t1 = Tensor::zeros(&[1, 2, 3, 4]);

        let t2 = t1.flatten_right(2);
        assert_eq!(t2.shape, vec![1, 24]);

        let t3 = t1.flatten_right(1);
        assert_eq!(t3.shape, vec![24]);
    }
}
