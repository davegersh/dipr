use super::Layer;
use crate::Tensor;

pub struct MeanPool {
    axis: usize,
    axis_size: Option<usize>,
}

impl MeanPool {
    pub fn new(axis: usize) -> Self {
        Self {
            axis,
            axis_size: None,
        }
    }
}

impl Layer for MeanPool {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut new_shape = x.shape.clone();

        new_shape.remove(self.axis);

        x.mean(self.axis).reshape(&new_shape)
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.axis_size = Some(x.shape[self.axis]);
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(size) = self.axis_size {
            let mut dj_dx = dj_dy / size as f32;

            let mut new_shape = dj_dx.shape.clone();

            new_shape.insert(self.axis, 1);
            dj_dx.reshape_mut(&new_shape);

            return dj_dx;
        }

        panic!(
            "Did not cache axis size before going backward for MeanPool layer! Make sure to run use forward_train()!"
        );
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}
