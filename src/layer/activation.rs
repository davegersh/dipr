use super::Layer;
use crate::Tensor;

pub struct ReLU {
    x_cache: Option<Tensor>,
    leak: f32,
}

impl ReLU {
    pub fn new(leak: f32) -> Self {
        Self {
            x_cache: None,
            leak,
        }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        x.map(|i| i.max(self.leak * i)) // y = max(leak * x, x)
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(x) = &self.x_cache {
            let dy_dx = x.map(|i| if i > 0.0 { 1.0 } else { self.leak });

            // chain rule!
            return dj_dy * dy_dx;
        }
        panic!("Input not cached when calculating gradient for ReLU Layer!");
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}

pub struct Sigmoid {
    y_cache: Option<Tensor>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { y_cache: None }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        x.map(|i| 1.0 / (1.0 + (-i).exp())) // y = sigmoid(x)
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        let y = self.forward(x);

        self.y_cache = Some(y.clone());
        y
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(y) = &self.y_cache {
            let dy_dx = y.map(|i| i * (1.0 - i));
            let dj_dx = dj_dy * dy_dx; // element wise (due to element-wise activation)

            return dj_dx;
        }
        panic!("Output not cached when calculating gradient for Sigmoid layer!");
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![]
    }

    fn zero_grad(&mut self) {}
}
