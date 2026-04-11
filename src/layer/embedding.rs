use super::Layer;
use crate::Tensor;

pub struct Embedding {
    pub weights: Tensor,
    pub weights_grad: Tensor,
    pub vocab_size: usize,
    pub embed_dim: usize,

    x_cache: Option<Tensor>,
}

impl Embedding {
    pub fn new(vocab_size: usize, embed_dim: usize) -> Self {
        let weights = Tensor::rand(&[vocab_size, embed_dim], 42);
        let weights_grad = Tensor::zeros(&[vocab_size, embed_dim]);

        Self {
            weights,
            weights_grad,
            vocab_size,
            embed_dim,
            x_cache: None,
        }
    }
}

impl Layer for Embedding {
    // Given a rank 2 tensor of token_IDs
    fn forward(&mut self, x: &Tensor) -> Tensor {
        assert_eq!(
            x.rank, 2,
            "Input shape for Embedding must be in the form of [batches, seq_length]!"
        );

        let batches = x.shape[0];
        let seq_length = x.shape[1];

        let mut output = Tensor::zeros(&[batches, seq_length, self.embed_dim]);

        for b in 0..batches {
            for s in 0..seq_length {
                let token = x[&[b, s]] as usize;

                for e in 0..self.embed_dim {
                    output[&[b, s, e]] = self.weights[&[token, e]];
                }
            }
        }

        output
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        self.x_cache = Some(x.clone());
        self.forward(x)
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        if let Some(x) = &self.x_cache {
            let batches = x.shape[0];
            let seq_length = x.shape[1];

            for b in 0..batches {
                for s in 0..seq_length {
                    let token = x[&[b, s]] as usize;

                    for e in 0..self.embed_dim {
                        self.weights_grad[&[token, e]] += dj_dy[&[b, s, e]];
                    }
                }
            }

            return Tensor::zeros(&x.shape);
        }

        panic!("Input not cached when calculating gradient for Embedding Layer!");
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![(&mut self.weights, &self.weights_grad)]
    }

    fn zero_grad(&mut self) {
        self.weights_grad.zero();
    }
}
