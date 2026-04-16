use super::Layer;
use crate::{Tensor, layer::activation::Softmax};

pub struct Attention {
    pub weights_q: Tensor,
    pub weights_k: Tensor,
    pub weights_v: Tensor,

    pub weights_q_grad: Tensor,
    pub weights_k_grad: Tensor,
    pub weights_v_grad: Tensor,

    softmax: Softmax,

    x_cache: Option<Tensor>,
    query_cache: Option<Tensor>,
    key_cache: Option<Tensor>,
    value_cache: Option<Tensor>,

    d_k: usize,
}

impl Attention {
    pub fn new(d_model: usize, d_k: usize) -> Self {
        let weights_shape = [d_model, d_k];

        Self {
            weights_q: Tensor::rand(&weights_shape, 42),
            weights_k: Tensor::rand(&weights_shape, 42),
            weights_v: Tensor::rand(&weights_shape, 42),
            weights_q_grad: Tensor::zeros(&weights_shape),
            weights_k_grad: Tensor::zeros(&weights_shape),
            weights_v_grad: Tensor::zeros(&weights_shape),
            softmax: Softmax::new(),
            x_cache: None,
            query_cache: None,
            key_cache: None,
            value_cache: None,
            d_k: d_k,
        }
    }

    fn compute(&mut self, x: &Tensor, is_training: bool) -> Tensor {
        let query = x.matmul(&self.weights_q); // Q = x @ w_q
        let key = x.matmul(&self.weights_k); // K = x @ w_k

        // S = (Q @ K^T) / sqrt(d)
        let score = query.matmul(&key.transpose()) / (self.d_k as f32).sqrt();

        let a = match is_training {
            true => self.softmax.forward_train(&score),
            false => self.softmax.forward(&score),
        };

        let value = x.matmul(&self.weights_v);

        let result = a.matmul(&value);

        if is_training {
            self.query_cache = Some(query);
            self.key_cache = Some(key);
            self.value_cache = Some(value);
        }

        result
    }
}

impl Layer for Attention {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        self.compute(x, false)
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.compute(x, true)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(a) = &self.softmax.y_cache {
            // gradient for value
            let dj_dv = a.transpose().matmul(dj_dy);

            if let Some(x) = &self.x_cache {
                let value = self.value_cache.as_ref().unwrap();

                // gradient for softmax output
                let dj_da = dj_dy.matmul(&value.transpose());

                // gradient for score
                let dj_ds = self.softmax.backward(&dj_da);

                // Recalculate query, key, and d (dims)
                let query = self.query_cache.as_ref().unwrap(); // Q = x @ w_q
                let key = self.key_cache.as_ref().unwrap(); // K = x @ w_k

                // gradient for query
                let dj_dq = dj_ds.matmul(&key) / (self.d_k as f32).sqrt();

                // gradient for key
                let dj_dk = dj_ds.transpose().matmul(&query) / (self.d_k as f32).sqrt();

                // gradients q, v, k weights
                self.weights_q_grad = x.transpose().matmul(&dj_dq).sum(0).flatten_left(2);
                self.weights_v_grad = x.transpose().matmul(&dj_dv).sum(0).flatten_left(2);
                self.weights_k_grad = x.transpose().matmul(&dj_dk).sum(0).flatten_left(2);

                // dj_dx combines q, v, k gradients and their multiplied weights
                return dj_dq.matmul(&self.weights_q.transpose())
                    + dj_dv.matmul(&self.weights_v.transpose())
                    + dj_dk.matmul(&self.weights_k.transpose());
            }
            panic!("Input not cached when calculating gradient for Attention Layer!");
        }
        panic!("Softmax output not cached when calculating gradient for Attention Layer!");
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![
            (&mut self.weights_q, &self.weights_q_grad),
            (&mut self.weights_k, &self.weights_k_grad),
            (&mut self.weights_v, &self.weights_v_grad),
        ]
    }

    fn zero_grad(&mut self) {
        self.weights_q_grad.zero();
        self.weights_k_grad.zero();
        self.weights_v_grad.zero();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_shape() {
        let mut attn = Attention::new(4, 4);
        let x = Tensor::ones(&[2, 2, 4]);

        let out = attn.forward_train(&x);

        assert_eq!(x.shape, out.shape);
    }

    #[test]
    fn test_attention_backward() {
        let mut attn = Attention::new(4, 4);
        let x = Tensor::iota(&[2, 2, 4]);

        let _output = attn.forward_train(&x);

        // dummy gradient
        let dl_dout = Tensor::ones(&[2, 2, 4]);

        let dl_dx = attn.backward(&dl_dout);

        // check gradient shape
        assert_eq!(dl_dx.shape, vec![2, 2, 4]);

        // Check gradients are non-zero
        let has_grad = dl_dx.data.iter().any(|&v| v.abs() > 1e-6);
        assert!(has_grad, "Gradients should be non-zero");
    }
}
