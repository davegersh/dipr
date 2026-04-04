use dipr::{
    Model, Tensor,
    layer::{Dense, Layer, WeightInit, activation::ReLU, activation::Sigmoid},
    loss::BinaryCrossEntropy,
    optim::SGD,
};

#[test]
fn test_xor_converge() {
    // create dataset
    let x = Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    // create model
    let mut model = Model::new(
        Box::new(SGD::new(1.00)),
        Box::new(BinaryCrossEntropy::new()),
    );

    model.add_layer(Dense::new(2, 5, WeightInit::Uniform));
    model.add_layer(ReLU::new(0.0));
    model.add_layer(Dense::new(5, 1, WeightInit::Uniform));
    model.add_layer(Sigmoid::new());

    // train it
    let history = model.train(&x, &y, 100);
    println!("\nCost History: {:?}\n", history);

    // check train convergence
    let final_preds = model.forward(&x);
    println!("Final: {:?}", final_preds);

    assert!(final_preds[&[0, 0]] < 0.05);
    assert!(final_preds[&[1, 0]] > 0.95);
    assert!(final_preds[&[2, 0]] > 0.95);
    assert!(final_preds[&[3, 0]] < 0.05);
}
