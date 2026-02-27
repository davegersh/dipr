use dipr::{
    Model, Tensor,
    layer::{Dense, Layer, ReLU, Sigmoid},
    loss::BinaryCrossEntropy,
    optim::SGD,
};

#[test]
fn xor_test_converge() {
    let x = Tensor::new(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![4, 1]);

    // create model
    let mut model = Model::new(
        Box::new(SGD::new(0.35)),
        Box::new(BinaryCrossEntropy::new()),
    );

    model.add_layer(Dense::new(&[3, 2]));
    model.add_layer(ReLU::new());
    model.add_layer(Dense::new(&[1, 3]));
    model.add_layer(Sigmoid::new());

    // train it
    model.train(&x, &y, 150);

    // test it
    let final_preds = model.forward(&x);
    println!("Final: {:?}", final_preds);

    assert!(final_preds[&[0, 0]] < 0.05);
    assert!(final_preds[&[1, 0]] > 0.95);
    assert!(final_preds[&[2, 0]] > 0.95);
    assert!(final_preds[&[3, 0]] < 0.05);
}
