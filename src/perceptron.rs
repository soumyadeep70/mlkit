
pub struct Perceptron {
    weights: Vec<f32>,
    bias: f32,
    step_threshold: f32,
}

impl Perceptron {
    pub fn new(n: usize, weights: Option<&[f32]>, bias: Option<f32>, step_threshold: Option<f32>) -> Self {
        let weights = match weights {
            Some(w) => {
                assert_eq!(w.len(), n, "weights must have length equal to n");
                w.to_vec()
            }
            None => vec![1.0; n],
        };

        Self {
            weights,
            bias: bias.unwrap_or(0.0),
            step_threshold: step_threshold.unwrap_or(0.5),
        }
    }

    pub fn train(&mut self, inputs: &[impl AsRef<[f32]>], targets: &[f32], learning_rate: f32, epochs: usize) {
        assert_eq!(inputs.len(), targets.len(), "inputs and targets must have the same length");

        for input in inputs {
            assert_eq!(input.as_ref().len(), self.weights.len(), "input dimension mismatch");
        }

        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let prediction = self.predict_one(input.as_ref());
                let error = *target - prediction;

                for (w, x) in self.weights.iter_mut().zip(input.as_ref().iter()) {
                    *w += learning_rate * error * x;
                }

                self.bias += learning_rate * error;
            }
        }
    }

    pub fn predict_one(&self, input: &[f32]) -> f32 {
        assert_eq!(input.len(), self.weights.len(), "input dimension mismatch");

        let sum = self.weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum::<f32>()+ self.bias;
        if sum >= self.step_threshold { 1.0 } else { 0.0 }
    }

    pub fn predict(&self, inputs: &[impl AsRef<[f32]>]) -> Vec<f32> {
        inputs.iter().map(|x| self.predict_one(x.as_ref())).collect()
    }

    pub fn num_features(&self) -> usize {
        self.weights.len()
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    pub fn bias(&self) -> f32 {
        self.bias
    }
}