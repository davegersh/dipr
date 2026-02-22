use std::ops::{Index, IndexMut};
use super::rand::XorShift;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f32>, //uses a 1D vector to handle any dimension
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub rank: usize
}

impl Tensor {
    fn shape_to_stride(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut cur_stride = 1;

        for dim in shape.iter().rev() {
            strides.push(cur_stride);
            cur_stride *= dim
        }

        strides.reverse();
        strides
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let stride = Self::shape_to_stride(&shape);
        let rank = shape.len();

        Self { data, shape, stride, rank }
    }


    pub fn zeros(shape: &[usize]) -> Self {
        let total_elements = shape.iter().product();
        let data = vec![0.0; total_elements];

        Self::new(data, shape.to_vec())
    }

    pub fn rand(shape: &[usize], seed: u32) -> Self {
        let total_elements = shape.iter().product();
        let data = XorShift::new(seed)
            .take(total_elements)
            .map(|x| x as f32)
            .collect();

        Self::new(data, shape.to_vec())
    }

    fn coords_to_index(&self, coords: &[usize]) -> usize {
        let mut index = 0;
        for i in 0..coords.len() {
            index += coords[i] * self.stride[i];
        }
        index
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
    fn zeros_test() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.data, vec![0., 0., 0., 0., 0., 0.]);
    }

    #[test]
    fn rand_test() {
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
    fn shape_to_stride_test() {
        // 2D
        let stride1 = Tensor::shape_to_stride(&[2, 3]);
        assert_eq!(stride1, vec![3, 1]);

        // 3D
        let stride2 = Tensor::shape_to_stride(&[2, 2, 3]);
        assert_eq!(stride2, vec![6, 3, 1]);
    }

    #[test]
    fn index_test() {
        let t = Tensor::new(vec![1.,2.,3.,4.,5.,6.], vec![2,3]);
        assert_eq!(t[&[1, 1]], t.data[4])
    }
}
