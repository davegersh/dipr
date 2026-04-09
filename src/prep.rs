use crate::Tensor;
use std::collections::HashMap;

pub struct OneHotEncoder {
    categories: Vec<String>,
    enc_map: HashMap<String, Vec<f32>>,
}

impl OneHotEncoder {
    pub fn new(categories: Vec<String>) -> Self {
        let mut enc_map = HashMap::new();

        let base_vec = vec![0f32; categories.len()];

        for (i, category) in categories.iter().enumerate() {
            let mut data = base_vec.clone();
            data[i] = 1.0;
            enc_map.insert(category.clone(), data);
        }

        Self {
            categories,
            enc_map,
        }
    }

    pub fn encode(&self, labels: &Vec<String>) -> Tensor {
        let mut data: Vec<f32> = Vec::new();

        for label in labels.iter() {
            if let Some(encoded) = self.enc_map.get(label) {
                data.extend(encoded);
            } else {
                panic!("\"{}\" is not in {:?}!", label, self.categories)
            }
        }

        Tensor::new(data, vec![labels.len(), self.categories.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encode() {
        let cats = vec![
            String::from("Red"),
            String::from("Green"),
            String::from("Blue"),
        ];

        let enc = OneHotEncoder::new(cats);

        let t = enc.encode(&vec![
            String::from("Blue"),
            String::from("Green"),
            String::from("Red"),
            String::from("Blue"),
        ]);

        let expected = Tensor::new(
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            vec![4, 3],
        );

        assert_eq!(t, expected);
    }
}
