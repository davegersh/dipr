use crate::Tensor;

pub trait Optimizer {
    fn update_parameter(&self, parameter: &mut Tensor, gradient: &Tensor);
}

pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn update_parameter(&self, parameter: &mut Tensor, gradient: &Tensor) {
        *parameter -= &(gradient * self.lr);
    }
}
