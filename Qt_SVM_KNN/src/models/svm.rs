use linfa::prelude::*;
use linfa_logistic::MultiLogisticRegression;
use ndarray::{Array1, Array2};

pub struct SVMModel {
    model: linfa_logistic::MultiFittedLogisticRegression<f64, usize>,
}

impl SVMModel {
    pub fn train(train: &DatasetBase<Array2<f64>, Array1<usize>>) -> Result<Self, Box<dyn std::error::Error>> {
        let model = MultiLogisticRegression::default()
            .max_iterations(100)
            .fit(train)?;
        Ok(SVMModel { model: model })
    }

    pub fn predict(&self, data: &Array2<f64>) -> Array1<usize> {
        self.model.predict(data)
    }

    pub fn evaluate_accuracy(&self, test: &DatasetBase<Array2<f64>, Array1<usize>>) -> f32 {
        let pred = self.predict(test.records());
        pred.iter()
            .zip(test.targets().iter())
            .filter(|(p, t)| p == t)
            .count() as f32 / test.nsamples() as f32
    }
}