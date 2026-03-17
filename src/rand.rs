use std::u32;

pub struct XorShift {
    pub seed: u32,
    pub state: u32,
    pub normalized: bool,
}

impl XorShift {
    pub fn new(seed: u32, normalized: bool) -> Self {
        XorShift {
            seed: seed,
            state: seed,
            normalized: normalized,
        }
    }

    pub fn shuffle<T>(&mut self, vec: &mut [T]) {
        let n = vec.len();
        for i in (1..n).rev() {
            let j = self.next().expect("AH") as usize % (i + 1);
            vec.swap(i, j);
        }
    }
}

impl Iterator for XorShift {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;

        self.state = x;

        let mut n = x as f64;

        if self.normalized {
            n /= u32::MAX as f64;
        }

        Some(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_test() {
        let r: Vec<f64> = XorShift::new(42, true).take(3).collect();
        assert_eq!(
            r,
            vec![
                0.0026438925421433273,
                0.6603119775327649,
                0.11095708681059933
            ]
        )
    }
}
