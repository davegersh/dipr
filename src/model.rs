use crate::Tensor;
use crate::layer::Layer;
use crate::loss::Loss;
use crate::optim::Optimizer;

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    loss: Box<dyn Loss>,
}

impl Model {
    pub fn new(optimizer: Box<dyn Optimizer>, loss: Box<dyn Loss>) -> Self {
        Self {
            layers: vec![],
            optimizer,
            loss,
        }
    }

    pub fn add_layer<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn train(&mut self, x_train: &Tensor, y_train: &Tensor, epochs: usize) -> Vec<f32> {
        let mut cost_history = vec![];

        for _ in 0..epochs {
            // forward propagation
            let y_pred = self.forward_train(x_train);

            // calculate cost
            let cost = self.loss.compute(y_train, &y_pred);
            cost_history.push(cost[&[0, 0]]);

            //backprop
            let dj_dy = self.loss.compute_derivative(y_train, &y_pred);
            self.backward(&dj_dy);

            // update parameters
            self.update_parameters();
        }

        cost_history
    }

    pub fn update_parameters(&mut self) {
        for layer in self.layers.iter_mut() {
            for (param, grad) in layer.parameters_mut() {
                self.optimizer.update_parameter(param, grad);
            }
        }
    }
}

impl Layer for Model {
    fn forward(&mut self, x: &Tensor) -> Tensor {
        let mut y = x.clone();
        for layer in self.layers.iter_mut() {
            y = layer.forward(&y);
        }
        y
    }

    fn forward_train(&mut self, x: &Tensor) -> Tensor {
        let mut y = x.clone();
        for layer in self.layers.iter_mut() {
            y = layer.forward_train(&y);
        }
        y
    }

    fn backward(&mut self, dj_dy: &Tensor) -> Tensor {
        let mut dj_dx = dj_dy.clone();
        for layer in self.layers.iter_mut().rev() {
            dj_dx = layer.backward(&dj_dx);
        }
        dj_dx
    }

    fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut params = vec![];

        for layer in self.layers.iter_mut() {
            params.extend(layer.parameters_mut());
        }

        params
    }

    fn zero_grad(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.zero_grad();
        }
    }
}
