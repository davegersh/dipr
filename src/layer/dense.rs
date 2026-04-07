use super::Layer;
use crate::Tensor;

pub enum WeightInit {
    Uniform,
    Xavier,
    He,
}

pub struct Dense {
    pub weights: Tensor,
    bias: Tensor,

    x_cache: Option<Tensor>,
    weights_grad: Tensor,
    bias_grad: Tensor,
}

impl Dense {
    pub fn new(inputs: usize, outputs: usize, init: WeightInit) -> Dense {
        let weights_shape = [inputs, outputs];

        let mut weights = Tensor::rand(&weights_shape, 42);

        let var = match init {
            WeightInit::Uniform => 1.0,
            WeightInit::He => 2.0 / (inputs as f32),
            WeightInit::Xavier => 2.0 / (inputs as f32 + outputs as f32),
        };

        weights *= var;

        let weights_grad = Tensor::zeros(&weights_shape);

        let bias_shape = [1, outputs];
        let bias = Tensor::zeros(&bias_shape);
        let bias_grad = bias.clone();

        Dense {
            weights,
            weights_grad,
            bias,
            bias_grad,
            x_cache: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        &x.matmul(&self.weights) + &self.bias //y = wx + b
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        self.zero_grad();

        if let Some(x) = &self.x_cache {
            self.weights_grad = x.transpose().matmul(dj_dy); // dj/dw = dj/dY * dy/dw
            self.bias_grad = dj_dy.sum(1); // dj/db = dj/dy * dy/db = dj/dy * 1

            // dj/dX = dj/dY * dY/dX = dj/dY * w
            return dj_dy.matmul(&self.weights.transpose());
        }

        panic!(
            "Input not cached when calculating gradient for Dense Layer! Did you run the forward_train() method?"
        );
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.weights, &self.weights_grad),
            (&mut self.bias, &self.bias_grad),
        ]
    }

    fn zero_grad(&mut self) {
        self.weights_grad.zero();
        self.bias_grad.zero();
    }
}
