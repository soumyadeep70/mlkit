pub mod perceptron;

use pyo3::prelude::*;
use crate::perceptron::Perceptron;

#[pymodule]
#[pyo3(name = "mlkit")]
fn mlkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPerceptron>()?;
    Ok(())
}


#[pyclass(name = "Perceptron")]
struct PyPerceptron {
    inner: Perceptron,
}

#[pymethods]
impl PyPerceptron {
    #[new]
    #[pyo3(signature = (n, weights=None, bias=None))]
    fn new(n: usize, weights: Option<Vec<f32>>, bias: Option<f32>) -> PyResult<Self> {
        Ok(Self {
            inner: Perceptron::new(
                n,
                weights.as_deref(),
                bias,
            ),
        })
    }

    #[pyo3(signature = (inputs, targets, learning_rate=0.1, epochs=100))]
    fn train(&mut self, inputs: Vec<Vec<f32>>, targets: Vec<f32>, learning_rate: f32, epochs: usize) {
        self.inner.train(&inputs, &targets, learning_rate, epochs);
    }

    fn predict_one(&self, input: Vec<f32>) -> f32 {
        self.inner.predict_one(&input)
    }

    fn predict(&self, inputs: Vec<Vec<f32>>) -> Vec<f32> {
        self.inner.predict(&inputs)
    }

    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights().to_vec()
    }

    #[getter]
    fn bias(&self) -> f32 {
        self.inner.bias()
    }

    #[getter]
    fn num_features(&self) -> usize {
        self.inner.num_features()
    }
}