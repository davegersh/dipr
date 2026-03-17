use std::fs::File;
use std::io::{BufRead, BufReader};

use dipr::loss::CategoricalCrossEntropy;
use dipr::prep::OneHotEncoder;
use dipr::rand::XorShift;
use dipr::{
    Model, Tensor,
    layer::{Dense, Layer, ReLU, Sigmoid},
    loss::BinaryCrossEntropy,
    optim::SGD,
};

fn load_data(path: &str) -> (Vec<f32>, Vec<String>) {
    let file = File::open(path).expect("iris.data file not found!");
    let reader = BufReader::new(file);

    let mut x_data = vec![];
    let mut y_data = vec![];

    for line in reader.lines() {
        let line = line.expect("Error reading line from iris.data!");

        let mut parts: Vec<&str> = line.split_terminator(',').collect();

        let target = parts.pop().expect("Couldn't find name!").to_owned();

        let f: Vec<f32> = parts[0..4]
            .iter_mut()
            .map(|x| x.parse::<f32>().unwrap())
            .collect();

        y_data.push(target);
        x_data.extend(f);
    }

    let mut x_rand = XorShift::new(42, false);
    let mut y_rand = XorShift::new(42, false);

    x_rand.shuffle(&mut x_data);
    y_rand.shuffle(&mut y_data);

    (x_data, y_data)
}

#[test]
fn test_iris_converge() {
    // load data
    let (x_data, y_data) = load_data("tests/iris_data/iris.data");

    let x = Tensor::new(x_data, vec![y_data.len(), 4]);

    // data prep
    let cats = vec![
        "setosa".to_owned(),
        "versicolor".to_owned(),
        "virginica".to_owned(),
    ];
    let enc = OneHotEncoder::new(cats);

    let y = enc.encode(&y_data);

    println!("x: {:?} \n y: {:?}", x, y);

    // create model
    let mut model = Model::new(
        Box::new(SGD::new(0.01)),
        Box::new(CategoricalCrossEntropy::new()),
    );

    model.add_layer(Dense::new(&[4, 16]));
    model.add_layer(ReLU::new());
    model.add_layer(Dense::new(&[16, 3]));

    // train it
    let history = model.train(&x, &y, 100);
    println!("\nCost History: {:?}\n", history);

    // test it
    let final_preds = model.forward(&x);
    println!("Final: {:?}", final_preds);

    assert_eq!(1.0, 0.0);
}
