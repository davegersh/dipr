use crate::tensor::Tensor;

trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, input_cache: &Tensor, grad: &Tensor) -> (Tensor, Self);
}

pub struct Dense {
    weights: Tensor,
    bias: Tensor,
}

impl Dense {
    pub fn new(shape: &[usize]) -> Dense {
        let weights = Tensor::rand(shape, 42);

        let mut bias_shape = shape.to_vec();
        bias_shape[weights.rank - 1] = 1;
        let bias = Tensor::zeros(&bias_shape);

        Dense {weights, bias }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.weights.matmul(input) + &self.bias
    }

    fn backward(&mut self, input_cache: &Tensor, d_output: &Tensor) -> (Tensor, Self) {
        let grads = Dense {
            weights: input_cache.transpose().matmul(d_output),
            bias: d_output.sum(0)
        };

        let d_input = d_output.matmul(&self.weights.transpose());

        (d_input, grads)
    }
}
