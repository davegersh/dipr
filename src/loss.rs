use crate::Tensor;

pub trait Loss {
    fn compute(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor;
    fn compute_derivative(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor;
}

pub struct LogLoss;

impl Loss for LogLoss {
    fn compute(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor {
        -y_truth * y_pred.map(|x| x.ln()) - (1.0 - y_truth) * y_pred.map(|x| (1.0 - x).ln())
    }

    fn compute_derivative(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor {
        -y_truth / y_pred + (1.0 - y_truth) / (1.0 - y_pred)
    }
}

pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for BinaryCrossEntropy {
    fn compute(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor {
        let log_loss = LogLoss {};
        let log_probs = log_loss.compute(y_truth, y_pred);
        log_probs.sum(1) / y_truth.shape[1] as f32
    }

    fn compute_derivative(&self, y_truth: &Tensor, y_pred: &Tensor) -> Tensor {
        let log_loss = LogLoss {};
        let log_probs = log_loss.compute_derivative(y_truth, y_pred);
        log_probs.sum(1) / y_truth.shape[1] as f32
    }
}
