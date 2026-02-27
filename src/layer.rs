use crate::tensor::Tensor;

// x => layer forward input
// y => layer forward output
// j => cost function
// dj_dy => partial derivative of j (cost) with respect to y (layer output)

/// Base trait for all layers in a model
pub trait Layer {
    /// Input tensor (x) => output tensor (y)
    fn forward(&mut self, x: &Tensor) -> Tensor;

    /// Same as forward(...) with added functionality to prepare data for backward(...)
    fn forward_train(&mut self, x: &Tensor) -> Tensor;

    /// Gradient from next layer => gradient of this layer
    fn backward(&mut self, dj_dy: &Tensor) -> Tensor;

    /// Returns a list of tuples containing the parameter and it's gradient respectively
    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)>;

    /// Resets the Layer's gradients to zero
    fn zero_grad(&mut self);
}

pub struct Dense {
    pub weights: Tensor,
    bias: Tensor,

    x_cache: Option<Tensor>,
    weights_grad: Tensor,
    bias_grad: Tensor,
}

impl Dense {
    pub fn new(shape: &[usize]) -> Dense {
        let weights = Tensor::rand(shape, 42);
        let weights_grad = Tensor::zeros(shape);

        let mut bias_shape = shape.to_vec();
        bias_shape[weights.rank - 1] = 1;
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
        // println!("Weights: {:?}", self.weights);
        x.matmul(&self.weights.transpose()) + &self.bias //y = wx + b
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        self.zero_grad();

        if let Some(x) = &self.x_cache {
            // println!("dj_dy: {:?}, x: {:?}", dj_dy.shape, x.shape);
            self.weights_grad = dj_dy.transpose().matmul(&x); // dj/dw = dj/dY * dy/dw
            self.bias_grad = dj_dy.sum(0); // dj/db = dj/dy * dy/db = dj/dy * 1

            // dj/dX = dj/dY * dY/dX = dj/dY * w
            // println!("weights: {:?}", self.weights.shape);
            return dj_dy.matmul(&self.weights);
        }

        panic!("Input not cached when calculating gradient for Dense Layer!");
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

pub struct ReLU {
    x_cache: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { x_cache: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        x.map(|i| i.max(0.0)) // y = max(0, x)
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(x) = &self.x_cache {
            let dy_dx = x.map(|i| if i > 0.0 { i } else { 0.0 });

            // chain rule!
            let dj_dx = dj_dy * dy_dx;

            return dj_dx;
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
        // println!("Sigmoid Input: {:?}, Output: {:?}", x, y);

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
