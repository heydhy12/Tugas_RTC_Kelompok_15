use ndarray::{Array, Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::error::Error;
use std::fs::File;
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

impl NeuralNetwork {
    pub fn new(input_size: usize, hidden_size1: usize, hidden_size2: usize, output_size: usize) -> Self {
        let he_std1 = (2.0 / input_size as f64).sqrt();
        let he_std2 = (2.0 / hidden_size1 as f64).sqrt();
        let he_std3 = (2.0 / hidden_size2 as f64).sqrt();

        let weights1 = Array::random((input_size, hidden_size1), Uniform::new(-he_std1, he_std1));
        let bias1 = Array::zeros(hidden_size1);

        let weights2 = Array::random((hidden_size1, hidden_size2), Uniform::new(-he_std2, he_std2));
        let bias2 = Array::zeros(hidden_size2);

        let weights3 = Array::random((hidden_size2, output_size), Uniform::new(-he_std3, he_std3));
        let bias3 = Array::zeros(output_size);

        NeuralNetwork {
            weights1,
            bias1,
            weights2,
            bias2,
            weights3,
            bias3,
            mean: Array::zeros(input_size),
            std: Array::ones(input_size),
        }
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let serialized = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model = bincode::deserialize(&buffer)?;
        Ok(model)
    }

    fn relu(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn relu_derivative(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    fn softmax(x: &Array2<f64>) -> Array2<f64> {
        let max_x = x.fold_axis(Axis(1), f64::NEG_INFINITY, |&a, &b| a.max(b));
        let exp_x = (x - &max_x.insert_axis(Axis(1))).mapv(f64::exp);
        let sum_exp_x = exp_x.sum_axis(Axis(1)).insert_axis(Axis(1));
        &exp_x / &sum_exp_x
    }

    fn cross_entropy_loss(y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let epsilon = 1e-10;
        -(y_true * y_pred.mapv(|v| (v + epsilon).ln())).sum()
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = Self::relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = Self::relu(&hidden_input2);

        let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
        Self::softmax(&output_input)
    }

    pub fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, learning_rate: f64, epochs: usize) 
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) 
    {
        let mut training_loss_history = Vec::new();
        let mut validation_loss_history = Vec::new();
        let mut training_accuracy_history = Vec::new();
        let mut validation_accuracy_history = Vec::new();

        self.mean = x.mean_axis(Axis(0)).unwrap();
        self.std = x.std_axis(Axis(0), 0.0);

        let x_normalized = (x - &self.mean) / &self.std;

        let split_idx = (x_normalized.shape()[0] as f64 * 0.8) as usize;
        let x_train = x_normalized.slice(s![..split_idx, ..]).to_owned();
        let y_train = y.slice(s![..split_idx, ..]).to_owned();
        let x_val = x_normalized.slice(s![split_idx.., ..]).to_owned();
        let y_val = y.slice(s![split_idx.., ..]).to_owned();

        for epoch in 0..epochs {
            let hidden_input1 = x_train.dot(&self.weights1) + &self.bias1;
            let hidden_output1 = Self::relu(&hidden_input1);

            let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
            let hidden_output2 = Self::relu(&hidden_input2);

            let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
            let output = Self::softmax(&output_input);

            let output_error = &output - &y_train;
            let hidden_error2 = output_error.dot(&self.weights3.t()) * Self::relu_derivative(&hidden_output2);
            let hidden_error1 = hidden_error2.dot(&self.weights2.t()) * Self::relu_derivative(&hidden_output1);

            self.weights3 -= &(learning_rate * hidden_output2.t().dot(&output_error));
            self.bias3 -= &(learning_rate * output_error.sum_axis(Axis(0)));

            self.weights2 -= &(learning_rate * hidden_output1.t().dot(&hidden_error2));
            self.bias2 -= &(learning_rate * hidden_error2.sum_axis(Axis(0)));

            self.weights1 -= &(learning_rate * x_train.t().dot(&hidden_error1));
            self.bias1 -= &(learning_rate * hidden_error1.sum_axis(Axis(0)));

            if epoch % 10 == 0 {
                let train_loss = Self::cross_entropy_loss(&output, &y_train);
                let train_accuracy = self.calculate_accuracy(&output, &y_train);
                
                let val_output = self.forward(&x_val);
                let val_loss = Self::cross_entropy_loss(&val_output, &y_val);
                let val_accuracy = self.calculate_accuracy(&val_output, &y_val);
                
                training_loss_history.push(train_loss);
                validation_loss_history.push(val_loss);
                training_accuracy_history.push(train_accuracy);
                validation_accuracy_history.push(val_accuracy);
            }

            if epoch % 100 == 0 {
                println!("Epoch: {}/{}", epoch, epochs);
            }
        }

        (training_loss_history, validation_loss_history, 
         training_accuracy_history, validation_accuracy_history)
    }

    pub fn calculate_accuracy(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
        let correct = y_pred
            .rows()
            .into_iter()
            .zip(y_true.rows().into_iter())
            .filter(|(pred_row, true_row)| {
                let predicted_class = pred_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let true_class = true_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                predicted_class == true_class
            })
            .count();

        correct as f64 / y_pred.shape()[0] as f64
    }

    pub fn evaluate(&self, x: &Array2<f64>, y: &Array2<f64>) -> (f64, f64) {
        let x_normalized = (x - &self.mean) / &self.std;
        let predictions = self.forward(&x_normalized);
        let loss = Self::cross_entropy_loss(&predictions, y);
        let accuracy = self.calculate_accuracy(&predictions, y);
        (loss, accuracy)
    }

    pub fn predict_single(&self, input: &[f64]) -> (String, Array1<f64>) {
        let input_array = Array::from_shape_vec((1, 4), input.to_vec()).unwrap();
        let normalized_input = (&input_array - &self.mean) / &self.std;
        
        let prediction = self.forward(&normalized_input);
        let probabilities = prediction.row(0).to_owned();
        
        let class_index = probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        
        let class_name = match class_index {
            0 => "Power Failure",
            1 => "Overstrain Failure",
            2 => "No Failure",
            3 => "Heat Dissipation Failure",
            _ => "Unknown",
        };
        
        (class_name.to_string(), probabilities)
    }
}