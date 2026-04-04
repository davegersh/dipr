pub mod dense;
pub use dense::Dense;
pub use dense::WeightInit;

pub mod activation;

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
