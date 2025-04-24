pub mod data;
pub mod models;
pub mod visualization;

use crate::models::svm::SVMModel;
use crate::data::loader::{load_data, normalize_data};
use linfa::dataset::Records;  // Untuk method nsamples()
use linfa::dataset::DatasetBase;
use ndarray::{Array1, Array2, s};
use serde::{Serialize, Deserialize};
use std::ffi::CString;


#[derive(Serialize, Deserialize)]
pub struct PredictionResult {
    pub svm_accuracy: f64,
    pub knn_accuracy: f64,
    pub sample_predictions: Vec<SamplePrediction>,
    pub svm_plot_path: String,
    pub knn_plot_path: String,
}

#[derive(Serialize, Deserialize)]
pub struct SamplePrediction {
    pub sample_id: usize,
    pub svm_prediction: String,
    pub knn_prediction: String,
    pub actual: String,
}

#[unsafe(no_mangle)]
pub extern "C" fn run_analysis(csv_path: *const libc::c_char) -> *mut libc::c_char {
    use std::ffi::{CString, CStr};
    
    let c_str = unsafe { CStr::from_ptr(csv_path) };
    let path = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    match analyze_data(path) {
        Ok(result) => {
            let json = serde_json::to_string(&result).unwrap();
            CString::new(json).unwrap().into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_string(s: *mut libc::c_char) {
    unsafe {
        if s.is_null() {
            return;
        }
        let _ = CString::from_raw(s); // Akan di-free ketika keluar scope
    }
}

pub fn analyze_data(file_path: &str) -> Result<PredictionResult, Box<dyn std::error::Error>> {
    use data::loader::{load_data, normalize_data};
    use models::{svm::SVMModel, knn::knn_predict_with_neighbors};
    use visualization::plotter::{plot_svm_neighbors, plot_knn_neighbors};
    use linfa::dataset::DatasetBase;
    use ndarray::s;

    // Load and prepare data
    let dataset = load_data(file_path)?;
    let normalized_features = normalize_data(dataset.features);
    let dataset = DatasetBase::new(normalized_features, dataset.labels);
    let (train, test) = dataset.split_with_ratio(0.8);

    // Train and evaluate SVM
    let svm_model = SVMModel::train(&train)?;
    let svm_accuracy = svm_model.evaluate_accuracy(&test) as f64;

    // Train and evaluate KNN
    let (knn_pred, neighbors) = knn_predict_with_neighbors(
        train.records(),
        train.targets(),
        test.records(),
        5
    );
    let knn_accuracy = evaluate_accuracy(&knn_pred, test.targets()) as f64;

    // Plot results
    let test_sample = test.records().slice(s![0..1, ..]).to_owned();
    let svm_pred = svm_model.predict(&test_sample);
    let svm_plot_path = "svm_neighbors.png".to_string();
    plot_svm_neighbors(
        train.records(),
        train.targets(),
        &test_sample,
        svm_pred[0],
        &svm_plot_path,
    )?;

    let knn_plot_path = "knn_neighbors.png".to_string();
    plot_knn_neighbors(
        train.records(),
        train.targets(),
        &test_sample,
        knn_pred[0],
        &neighbors[0],
        &knn_plot_path,
    )?;

    // Prepare sample predictions
    let sample_predictions = get_sample_predictions(&train, &test, &svm_model, &knn_pred)?;

    Ok(PredictionResult {
        svm_accuracy,
        knn_accuracy,
        sample_predictions,
        svm_plot_path,
        knn_plot_path,
    })
}

fn evaluate_accuracy(predictions: &Array1<usize>, actual: &Array1<usize>) -> f64 {
    let correct = predictions
        .iter()
        .zip(actual.iter())
        .filter(|(pred, actual)| pred == actual)
        .count();
    correct as f64 / actual.len() as f64
}

fn get_sample_predictions(
    train: &DatasetBase<Array2<f64>, Array1<usize>>,
    test: &DatasetBase<Array2<f64>, Array1<usize>>,
    svm_model: &SVMModel,
    knn_predictions: &Array1<usize>,
) -> Result<Vec<SamplePrediction>, Box<dyn std::error::Error>> {
    use rand::{thread_rng, seq::SliceRandom};
    
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..test.nsamples()).collect();
    indices.shuffle(&mut rng);
    let selected_indices = &indices[0..10.min(indices.len())];

    let test_samples = test.records().select(ndarray::Axis(0), selected_indices);
    let test_labels = test.targets().select(ndarray::Axis(0), selected_indices);

    let svm_predictions = svm_model.predict(&test_samples);
    let knn_selected = knn_predictions.select(ndarray::Axis(0), selected_indices);

    let mut sample_predictions = Vec::new();
    for (i, ((svm_pred, knn_pred), actual)) in svm_predictions.iter()
        .zip(knn_selected.iter())
        .zip(test_labels.iter())
        .enumerate()
    {
        sample_predictions.push(SamplePrediction {
            sample_id: i + 1,
            svm_prediction: get_failure_type(*svm_pred).to_string(),
            knn_prediction: get_failure_type(*knn_pred).to_string(),
            actual: get_failure_type(*actual).to_string(),
        });
    }

    Ok(sample_predictions)
}

fn get_failure_type(label: usize) -> &'static str {
    match label {
        0 => "No Failure",
        1 => "Heat Dissipation Failure",
        2 => "Overstrain Failure",
        3 => "Power Failure",
        _ => "Unknown",
    }
}