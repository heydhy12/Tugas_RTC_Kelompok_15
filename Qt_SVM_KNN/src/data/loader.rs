use ndarray::{Array1, Array2};
use csv::ReaderBuilder;
use std::fs::File;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct LoadedData {
    pub features: Array2<f64>,
    pub labels: Array1<usize>,
}

pub fn load_data(file_path: &str) -> Result<LoadedData, Box<dyn std::error::Error>> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
    let mut rng = thread_rng();
    records.as_mut_slice().shuffle(&mut rng);

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for record in &records {
        let feat: Vec<f64> = (0..4)
            .map(|i| record[i].trim().parse::<f64>().unwrap_or(0.0))
            .collect();
        features.push(feat);

        let label = match record[4].trim().to_lowercase().as_str() {
            "no failure" => 0,
            "heat dissipation failure" => 1,
            "overstrain failure" => 2,
            "power failure" => 3,
            _ => 0,
        };
        labels.push(label);
    }

    let features = Array2::from_shape_vec((features.len(), 4), features.concat())?;
    let labels = Array1::from_vec(labels);

    Ok(LoadedData { features, labels })
}

pub fn normalize_data(mut data: Array2<f64>) -> Array2<f64> {
    for mut col in data.axis_iter_mut(ndarray::Axis(1)) {
        let mean = col.mean().unwrap();
        let std = col.std(0.0);
        col.iter_mut().for_each(|x| *x = (*x - mean) / std);
    }
    data
}