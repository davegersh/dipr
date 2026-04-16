use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use dipr::layer::attention::Attention;
use dipr::layer::embedding::Embedding;
use dipr::layer::shapes::MeanPool;
use dipr::loss::CategoricalCrossEntropy;
use dipr::prep::OneHotEncoder;
use dipr::rand::XorShift;
use dipr::{
    Model, Tensor,
    layer::{Dense, Layer, WeightInit},
    optim::SGD,
};

pub fn read_data(path: &str) -> (Vec<String>, Vec<String>) {
    let file = File::open(path).expect("Sentiment data file not found!");
    let reader = BufReader::new(file);

    let mut sentences: Vec<String> = vec![];
    let mut sentiments: Vec<String> = vec![];

    for line in reader.lines() {
        let line = line.expect("Error reading line from sentiment data!");

        let parts: Vec<&str> = line.split_terminator(',').collect();

        let sentence = parts[0].to_lowercase();

        sentences.push(sentence);
        sentiments.push(parts[1].to_owned());
    }

    (sentences, sentiments)
}

// For now, we tokenize by just taking the words of a sentence (separate fn in case this changes!)
pub fn tokenize(sentence: &str) -> impl Iterator<Item = &str> {
    sentence.split_whitespace()
}

pub fn extract_vocab(sentences: &Vec<String>) -> HashMap<String, usize> {
    let mut vocab = HashMap::new();

    vocab.insert("<PAD>".to_owned(), 0);
    let mut cur_id = 1;

    for sentence in sentences.iter() {
        let tokens = tokenize(sentence);

        for token in tokens {
            if !vocab.contains_key(token) {
                vocab.insert(token.to_owned(), cur_id);
                cur_id += 1;
            }
        }
    }

    vocab
}

pub fn sentence_to_token_ids(sentences: &Vec<String>, vocab: &HashMap<String, usize>) -> Tensor {
    let mut seqs: Vec<Vec<usize>> = vec![];

    let mut longest_seq = 0;

    // convert sentences into token ids
    for sentence in sentences.iter() {
        let tokens = tokenize(sentence);

        let mut seq = vec![];

        for token in tokens {
            seq.push(vocab[token]);
        }

        longest_seq = longest_seq.max(seq.len());

        seqs.push(seq);
    }

    // pad the sequences that are shorter than the longest sequence
    for seq in seqs.iter_mut() {
        while seq.len() < longest_seq {
            seq.push(0);
        }
    }

    let mut token_id_data: Vec<f32> = vec![];
    for seq in seqs {
        token_id_data.extend(seq.iter().map(|f| *f as f32));
    }

    // Convert seqs to a tensor
    Tensor::new(token_id_data, vec![sentences.len(), longest_seq])
}

#[test]
fn test_sentiment_converge() {
    let (mut sentences, mut sentiments) = read_data("tests/data/sentiment.data");

    println!("Sentences: {:?}\n Sentiments: {:?}", sentences, sentiments);

    let vocab = extract_vocab(&sentences);

    println!("Vocab: {:?}", vocab);
    // data shuffling
    let mut x_rand = XorShift::new(42, false);
    x_rand.shuffle(&mut sentences);

    let x = sentence_to_token_ids(&sentences, &vocab);
    println!("Token IDs: {:?}", x);

    let mut y_rand = XorShift::new(42, false);
    y_rand.shuffle(&mut sentiments);

    let sentiment_encoder = OneHotEncoder::new(vec!["positive".to_owned(), "negative".to_owned()]);

    let y = sentiment_encoder.encode(&sentiments);

    println!("{:?}", y);

    // create model
    let mut model = Model::new(
        Box::new(SGD::new(0.05)),
        Box::new(CategoricalCrossEntropy::new()),
    );

    model.add_layer(Embedding::new(vocab.len(), 16));
    model.add_layer(Attention::new(16, 16));
    model.add_layer(MeanPool::new(1));
    model.add_layer(Dense::new(16, 2, WeightInit::Xavier));

    println!("Total Model Params: {:?}", model.calculate_total_params());

    // train it
    let history = model.train(&x, &y, 500);
    println!("\nCost History: {:?}\n", history);

    // check train convergence
    let output = model.forward(&x);

    println!("Output: {:?}", output);

    assert!(false);
}
