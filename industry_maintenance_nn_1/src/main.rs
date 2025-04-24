use csv::Reader;
use ndarray::{Array, Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fs::File;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::Instant;
use chrono::Duration;
use serde::{Serialize, Deserialize};
use std::io::{Write, Read};
use std::io;

#[derive(Debug, serde::Deserialize)]
struct Record {
    air_temperature_k: f64,
    process_temperature_k: f64,
    rotational_speed_rpm: f64,
    torque_nm: f64,
    failure_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct NeuralNetwork {
    weights1: Array2<f64>,
    bias1: Array1<f64>,
    weights2: Array2<f64>,
    bias2: Array1<f64>,
    weights3: Array2<f64>,
    bias3: Array1<f64>,
    mean: Array1<f64>,
    std: Array1<f64>,
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_size1: usize, hidden_size2: usize, output_size: usize) -> Self {
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

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let serialized = bincode::serialize(self)?;
        let mut file = File::create(path)?;
        file.write_all(&serialized)?;
        Ok(())
    }

    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
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

    fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = Self::relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = Self::relu(&hidden_input2);

        let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
        Self::softmax(&output_input)
    }

    fn train(&mut self, x: &Array2<f64>, y: &Array2<f64>, learning_rate: f64, epochs: usize) 
        -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) 
    {
        let mut training_loss_history = Vec::new();
        let mut validation_loss_history = Vec::new();
        let mut training_accuracy_history = Vec::new();
        let mut validation_accuracy_history = Vec::new();

        // Calculate and store normalization parameters
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

    fn calculate_accuracy(&self, y_pred: &Array2<f64>, y_true: &Array2<f64>) -> f64 {
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

    fn evaluate(&self, x: &Array2<f64>, y: &Array2<f64>) -> (f64, f64) {
        let x_normalized = (x - &self.mean) / &self.std;
        let predictions = self.forward(&x_normalized);
        let loss = Self::cross_entropy_loss(&predictions, y);
        let accuracy = self.calculate_accuracy(&predictions, y);
        (loss, accuracy)
    }

    fn predict_single(&self, input: &[f64]) -> (String, Array1<f64>) {
        // Convert input to array and normalize
        let input_array = Array::from_shape_vec((1, 4), input.to_vec()).unwrap();
        let normalized_input = (&input_array - &self.mean) / &self.std;
        
        // Get prediction probabilities
        let prediction = self.forward(&normalized_input);
        let probabilities = prediction.row(0).to_owned();
        
        // Get the predicted class
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

fn plot_metrics(
    training_loss: &[f64],
    validation_loss: &[f64],
    training_accuracy: &[f64],
    validation_accuracy: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    let root = BitMapBackend::new("metrics_plot.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    // Plot Loss
    let max_steps = training_loss.len().max(validation_loss.len());
    let max_loss = training_loss.iter()
        .chain(validation_loss.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart_loss = ChartBuilder::on(&upper)
        .caption("Training & Validation Loss", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_steps, 0.0..max_loss)?;

    chart_loss.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    chart_loss.draw_series(LineSeries::new(
        training_loss.iter().enumerate().map(|(x, y)| (x, *y)),
        &RED,
    ))?.label("Training Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_loss.draw_series(LineSeries::new(
        validation_loss.iter().enumerate().map(|(x, y)| (x, *y)),
        &BLUE,
    ))?.label("Validation Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_loss.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // Plot Accuracy
    let max_acc = training_accuracy.iter()
        .chain(validation_accuracy.iter())
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart_acc = ChartBuilder::on(&lower)
        .caption("Training & Validation Accuracy", ("sans-serif", 25))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(0..max_steps, 0.0..max_acc)?;

    chart_acc.configure_mesh()
        .x_desc("Epoch")
        .y_desc("Accuracy")
        .draw()?;

    chart_acc.draw_series(LineSeries::new(
        training_accuracy.iter().enumerate().map(|(x, y)| (x, *y)),
        &GREEN,
    ))?.label("Training Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart_acc.draw_series(LineSeries::new(
        validation_accuracy.iter().enumerate().map(|(x, y)| (x, *y)),
        &MAGENTA,
    ))?.label("Validation Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart_acc.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn get_user_input(prompt: &str) -> f64 {
    loop {
        println!("{}", prompt);
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");
        
        match input.trim().parse() {
            Ok(num) => return num,
            Err(_) => println!("Please enter a valid number!"),
        }
    }
}

fn manual_prediction(nn: &NeuralNetwork) {
    println!("\nManual Prediction Mode");
    println!("---------------------");
    
    let air_temp = get_user_input("Enter air temperature (K):");
    let process_temp = get_user_input("Enter process temperature (K):");
    let rotational_speed = get_user_input("Enter rotational speed (rpm):");
    let torque = get_user_input("Enter torque (Nm):");
    
    let input = vec![air_temp, process_temp, rotational_speed, torque];
    let (predicted_class, probabilities) = nn.predict_single(&input);
    
    println!("\nPrediction Results:");
    println!("------------------");
    println!("Most likely failure type: {}", predicted_class);
    println!("\nProbability distribution:");
    println!("Power Failure: {:.2}%", probabilities[0] * 100.0);
    println!("Overstrain Failure: {:.2}%", probabilities[1] * 100.0);
    println!("No Failure: {:.2}%", probabilities[2] * 100.0);
    println!("Heat Dissipation Failure: {:.2}%", probabilities[3] * 100.0);
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Membaca file CSV...");
    let file_path = "csv/industry_maintenance.csv";
    let file = File::open(file_path).expect("Gagal membuka file CSV!");
    let mut rdr = Reader::from_reader(file);

    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    if records.is_empty() {
        eprintln!("Tidak ada data yang dibaca dari file CSV!");
        return Ok(());
    }
    println!("Berhasil membaca {} records.", records.len());

    records.shuffle(&mut thread_rng());

    let mut x = Array2::zeros((records.len(), 4));
    let mut y = Array2::zeros((records.len(), 4));

    for (i, record) in records.iter().enumerate() {
        x[[i, 0]] = record.air_temperature_k;
        x[[i, 1]] = record.process_temperature_k;
        x[[i, 2]] = record.rotational_speed_rpm;
        x[[i, 3]] = record.torque_nm;

        match record.failure_type.as_str() {
            "Power Failure" => y[[i, 0]] = 1.0,
            "Overstrain Failure" => y[[i, 1]] = 1.0,
            "No Failure" => y[[i, 2]] = 1.0,
            "Heat Dissipation Failure" => y[[i, 3]] = 1.0,
            _ => (),
        }
    }

    //PARAMETER TRAINING
    println!("Inisialisasi neural network...");
    let input_size = 4;
    let hidden_size1 = 5;
    let hidden_size2 = 3;
    let output_size = 4;
    let learning_rate = 0.001;
    let epochs = 1000;

    let mut nn = NeuralNetwork::new(input_size, hidden_size1, hidden_size2, output_size);

    println!("Memulai training...");
    let start_time = Instant::now();
    let (train_loss, val_loss, train_accuracy, val_accuracy) = nn.train(&x, &y, learning_rate, epochs);
    let duration = Duration::from_std(start_time.elapsed()).unwrap();
    println!("Training selesai dalam {} detik!", duration.num_seconds());
    println!("\nAkurasi Training: {:.2}%", train_accuracy.last().unwrap() * 100.0);
    println!("Akurasi Validasi: {:.2}%", val_accuracy.last().unwrap() * 100.0);
    
    // Save the trained model
    let model_path = "Trained_model.bin";
    nn.save(model_path)?;
    println!("\nModel berhasil disimpan sebagai: {}", model_path);

    // Load the trained model to demonstrate usage of `load`
    let loaded_nn = NeuralNetwork::load(model_path)?;
    println!("Model berhasil dimuat kembali dari: {}", model_path);

    println!("Membuat plot training progress...");
    plot_metrics(&train_loss, &val_loss, &train_accuracy, &val_accuracy)?;
    println!("Plot berhasil disimpan sebagai training_metrics.png");

    println!("\nEvaluasi Model:");
    let (train_loss, train_accuracy) = loaded_nn.evaluate(&x, &y);
    println!("Training Data - Loss: {:.4}, Akurasi: {:.2}%", train_loss, train_accuracy * 100.0);

    // Manual prediction loop
    loop {
        println!("\nMenu:");
        println!("1. Lakukan prediksi manual");
        println!("2. Keluar");
        
        let mut choice = String::new();
        io::stdin().read_line(&mut choice).expect("Failed to read line");
        
        match choice.trim() {
            "1" => manual_prediction(&loaded_nn),
            "2" => break,
            _ => println!("Pilihan tidak valid!"),
        }
    }

    Ok(())
}