use csv::Reader;
use ndarray::{Array, Array1, Array2, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::fs::File;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Serialize, Deserialize};
use plotters::prelude::*;
use std::os::raw::{c_void, c_int, c_float, c_char};
use std::ffi::{CStr, CString};

#[repr(C)]
pub struct TrainingProgress {
    pub epoch: c_int,
    pub training_accuracy: c_float,
    pub validation_accuracy: c_float,
    pub training_loss: c_float,
    pub validation_loss: c_float,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub epochs: Vec<usize>,
    pub training_accuracies: Vec<f64>,
    pub validation_accuracies: Vec<f64>,
    pub training_losses: Vec<f64>,
    pub validation_losses: Vec<f64>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        TrainingHistory {
            epochs: Vec::new(),
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
            training_losses: Vec::new(),
            validation_losses: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainedModel {
    pub network: NeuralNetwork,
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
    pub final_accuracy: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub weights1: Array2<f64>,
    pub bias1: Array1<f64>,
    pub weights2: Array2<f64>,
    pub bias2: Array1<f64>,
    pub weights3: Array2<f64>,
    pub bias3: Array1<f64>,
}

#[repr(C)]
pub struct PredictionProgress {
    pub input_data: [c_float; 4],
    pub probabilities: [c_float; 4],
    pub predicted_class: c_int,
}

#[repr(C)]
#[derive(Debug, Serialize)]
pub struct PredictionResult {
    pub class_id: i32,
    pub probabilities: [f32; 4],
    #[serde(skip)]
    pub class_name: *mut c_char,
}

impl Default for PredictionResult {
    fn default() -> Self {
        Self {
            class_id: 0,
            probabilities: [0.0; 4],
            class_name: std::ptr::null_mut(),
        }
    }
}

struct TrainingProgressReport {
    epoch: usize,
    training_accuracy: f64,
    validation_accuracy: f64,
    training_loss: f64,
    validation_loss: f64,
}

fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn free_c_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

fn train_network_with_progress<F>(
    csv_path: &str,
    epochs: usize,
    plot_path: &str,
    progress_callback: F,
) -> Result<TrainedModel, Box<dyn Error>>
where
    F: Fn(TrainingProgressReport),
{
    let file = File::open(csv_path)?;
    let mut rdr = Reader::from_reader(file);
    let mut records = Vec::new();

    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

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

    let mean = x.mean_axis(Axis(0)).unwrap();
    let std = x.std_axis(Axis(0), 1.0);
    let x_normalized = (x - &mean) / &std;

    let split_idx = (x_normalized.shape()[0] as f64 * 0.8) as usize;
    let x_train = x_normalized.slice(s![..split_idx, ..]).to_owned();
    let y_train = y.slice(s![..split_idx, ..]).to_owned();
    let x_val = x_normalized.slice(s![split_idx.., ..]).to_owned();
    let y_val = y.slice(s![split_idx.., ..]).to_owned();

    let input_size = 4;
    let hidden_size1 = 5;
    let hidden_size2 = 3;
    let output_size = 4;
    let initial_learning_rate = 0.0005; 
    let lambda = 0.1; 
    let batch_size = 32; 

    let mut nn = NeuralNetwork::new(input_size, hidden_size1, hidden_size2, output_size);
    let mut history = TrainingHistory::default();

    for epoch in 0..epochs {
        let current_learning_rate = initial_learning_rate * 0.95f64.powf(epoch as f64 / 100.0);
        
        // Mini-batch training
        let num_samples = x_train.shape()[0];
        let mut indices: Vec<usize> = (0..num_samples).collect();
        indices.shuffle(&mut thread_rng());
        
        for batch_indices in indices.chunks(batch_size) {
            let x_batch = x_train.select(Axis(0), batch_indices);
            let y_batch = y_train.select(Axis(0), batch_indices);
            nn.train(&x_batch, &y_batch, current_learning_rate, lambda);
        }
        
        let train_output = nn.forward(&x_train);
        let train_loss = cross_entropy_loss(&train_output, &y_train);
        let train_accuracy = calculate_accuracy(&y_train, &train_output);
        
        let val_output = nn.forward(&x_val);
        let val_loss = cross_entropy_loss(&val_output, &y_val);
        let val_accuracy = calculate_accuracy(&y_val, &val_output);
        
        history.epochs.push(epoch);
        history.training_accuracies.push(train_accuracy);
        history.validation_accuracies.push(val_accuracy);
        history.training_losses.push(train_loss);
        history.validation_losses.push(val_loss);
        
        let progress = TrainingProgressReport {
            epoch,
            training_accuracy: train_accuracy,
            validation_accuracy: val_accuracy,
            training_loss: train_loss,
            validation_loss: val_loss,
        };
        
        println!(
            "Epoch {}/{} - Train Loss: {:.4}, Val Loss: {:.4} | Train Acc: {:.2}%, Val Acc: {:.2}%",
            epoch + 1,
            epochs,
            progress.training_loss,
            progress.validation_loss,
            progress.training_accuracy * 100.0,
            progress.validation_accuracy * 100.0
        );
        
        progress_callback(progress);
    }

    let final_output = nn.forward(&x_val);
    let final_accuracy = calculate_accuracy(&y_val, &final_output);

    create_smooth_plot(&history, plot_path)?;

    Ok(TrainedModel {
        network: nn,
        mean,
        std,
        final_accuracy,
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn train_industry_model_with_progress(
    csv_path: *const c_char,
    epochs: c_int,
    plot_path: *const c_char,
    model_path: *const c_char,
    accuracy: *mut f64,
    progress_callback: extern "C" fn(*mut c_void, TrainingProgress),
    callback_context: *mut c_void,
) -> bool {
    let csv_path_str = unsafe { CStr::from_ptr(csv_path).to_str().unwrap() };
    let plot_path_str = unsafe { CStr::from_ptr(plot_path).to_str().unwrap() };
    let model_path_str = unsafe { CStr::from_ptr(model_path).to_str().unwrap() };

    match train_network_with_progress(
        csv_path_str,
        epochs as usize,
        plot_path_str,
        |progress| {
            let c_progress = TrainingProgress {
                epoch: progress.epoch as c_int,
                training_accuracy: progress.training_accuracy as c_float,
                validation_accuracy: progress.validation_accuracy as c_float,
                training_loss: progress.training_loss as c_float,
                validation_loss: progress.validation_loss as c_float,
            };
            progress_callback(callback_context, c_progress);
        }
    ) {
        Ok(model) => {
            unsafe { *accuracy = model.final_accuracy };
            if let Ok(model_data) = bincode::serialize(&model) {
                std::fs::write(model_path_str, model_data).is_ok()
            } else {
                eprintln!("Failed to serialize model");
                false
            }
        }
        Err(e) => {
            eprintln!("Training error: {}", e);
            false
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_failure_type_with_progress(
    air_temp: f32,
    process_temp: f32,
    rotational_speed: f32,
    torque: f32,
    model_path: *const c_char,
    progress_callback: extern "C" fn(*mut c_void, PredictionProgress),
    callback_context: *mut c_void,
) -> *mut PredictionResult {
    let model_path_str = match unsafe { CStr::from_ptr(model_path).to_str() } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error converting model path: {}", e);
            return std::ptr::null_mut();
        }
    };

    let model_data = match std::fs::read(model_path_str) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error reading model file: {}", e);
            return std::ptr::null_mut();
        }
    };

    let trained_model: TrainedModel = match bincode::deserialize(&model_data) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error deserializing model: {}", e);
            return std::ptr::null_mut();
        }
    };

    let input = Array::from_shape_vec((1, 4), vec![
        air_temp as f64,
        process_temp as f64,
        rotational_speed as f64,
        torque as f64
    ]).unwrap();

    let normalized_input = (&input - &trained_model.mean) / &trained_model.std;
    let prediction = trained_model.network.forward(&normalized_input);

    let mut prob_array = [0.0f32; 4];
    for (i, &prob) in prediction.row(0).iter().take(4).enumerate() {
        prob_array[i] = prob as f32;
    }

    let class_index = prediction.row(0)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as i32)
        .unwrap_or(0);

    let class_name = match class_index {
        0 => "Power Failure",
        1 => "Overstrain Failure",
        2 => "No Failure",
        3 => "Heat Dissipation Failure",
        _ => "Unknown",
    };

    let progress = PredictionProgress {
        input_data: [air_temp, process_temp, rotational_speed, torque],
        probabilities: prob_array,
        predicted_class: class_index,
    };
    progress_callback(callback_context, progress);

    println!(
        "| Prediction \n| Input: [{:.2}, {:.2}, {:.2}, {:.2}] \n| Probabilities: [{:.2}%, {:.2}%, {:.2}%, {:.2}%] \n| Predicted: {}",
        air_temp, process_temp, rotational_speed, torque,
        prob_array[0] * 100.0,
        prob_array[1] * 100.0,
        prob_array[2] * 100.0,
        prob_array[3] * 100.0,
        class_name
    );

    Box::into_raw(Box::new(PredictionResult {
        class_id: class_index,
        probabilities: prob_array,
        class_name: to_c_string(class_name),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn free_prediction_result(result: *mut PredictionResult) {
    if !result.is_null() {
        unsafe {
            free_c_string((*result).class_name);
            drop(Box::from_raw(result));
        }
    }
}

fn calculate_accuracy(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
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

fn create_smooth_plot(history: &TrainingHistory, path: &str) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let (upper, lower) = root.split_vertically(400);

    // Smoothing data with moving average
    let smooth_training_losses: Vec<f64> = history.training_losses.windows(5)
        .map(|w| w.iter().sum::<f64>() / 5.0)
        .collect();
    let smooth_val_losses: Vec<f64> = history.validation_losses.windows(5)
        .map(|w| w.iter().sum::<f64>() / 5.0)
        .collect();
    let smooth_training_acc: Vec<f64> = history.training_accuracies.windows(5)
        .map(|w| w.iter().sum::<f64>() / 5.0)
        .collect();
    let smooth_val_acc: Vec<f64> = history.validation_accuracies.windows(5)
        .map(|w| w.iter().sum::<f64>() / 5.0)
        .collect();

    let max_steps = history.epochs.len();
    let max_loss = smooth_training_losses.iter()
        .chain(smooth_val_losses.iter())
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
        history.epochs.iter().take(smooth_training_losses.len()).zip(smooth_training_losses.iter()).map(|(x, y)| (*x, *y)),
        &RED,
    ))?.label("Training Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_loss.draw_series(LineSeries::new(
        history.epochs.iter().take(smooth_val_losses.len()).zip(smooth_val_losses.iter()).map(|(x, y)| (*x, *y)),
        &BLUE,
    ))?.label("Validation Loss")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart_loss.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    let max_acc = smooth_training_acc.iter()
        .chain(smooth_val_acc.iter())
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
        history.epochs.iter().take(smooth_training_acc.len()).zip(smooth_training_acc.iter()).map(|(x, y)| (*x, *y)),
        &GREEN,
    ))?.label("Training Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart_acc.draw_series(LineSeries::new(
        history.epochs.iter().take(smooth_val_acc.len()).zip(smooth_val_acc.iter()).map(|(x, y)| (*x, *y)),
        &MAGENTA,
    ))?.label("Validation Accuracy")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &MAGENTA));

    chart_acc.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
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

fn cross_entropy_loss(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
    let epsilon = 1e-10;
    -(y_true * y_pred.mapv(|v| (v + epsilon).ln())).sum()
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
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
        softmax(&output_input)
    }

    pub fn train(
        &mut self, 
        x: &Array2<f64>, 
        y: &Array2<f64>, 
        learning_rate: f64, 
        lambda: f64
    ) {
        let hidden_input1 = x.dot(&self.weights1) + &self.bias1;
        let hidden_output1 = relu(&hidden_input1);

        let hidden_input2 = hidden_output1.dot(&self.weights2) + &self.bias2;
        let hidden_output2 = relu(&hidden_input2);

        let output_input = hidden_output2.dot(&self.weights3) + &self.bias3;
        let output = softmax(&output_input);

        let output_error = &output - y;
        let hidden_error2 = output_error.dot(&self.weights3.t()) * relu_derivative(&hidden_output2);
        let hidden_error1 = hidden_error2.dot(&self.weights2.t()) * relu_derivative(&hidden_output1);

        self.weights3 -= &(learning_rate * (hidden_output2.t().dot(&output_error) + lambda * &self.weights3));
        self.bias3 -= &(learning_rate * output_error.sum_axis(Axis(0)));

        self.weights2 -= &(learning_rate * (hidden_output1.t().dot(&hidden_error2) + lambda * &self.weights2));
        self.bias2 -= &(learning_rate * hidden_error2.sum_axis(Axis(0)));

        self.weights1 -= &(learning_rate * (x.t().dot(&hidden_error1) + lambda * &self.weights1));
        self.bias1 -= &(learning_rate * hidden_error1.sum_axis(Axis(0)));
    }
}

#[derive(Debug, serde::Deserialize)]
struct Record {
    air_temperature_k: f64,
    process_temperature_k: f64,
    rotational_speed_rpm: f64,
    torque_nm: f64,
    failure_type: String,
}