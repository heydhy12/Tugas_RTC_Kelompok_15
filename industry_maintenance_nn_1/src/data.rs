use csv::Reader;
use ndarray::Array2;
use serde::Deserialize;
use std::fs::File;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Debug, Deserialize)]
pub struct Record {
    pub air_temperature_k: f64,
    pub process_temperature_k: f64,
    pub rotational_speed_rpm: f64,
    pub torque_nm: f64,
    pub failure_type: String,
}

pub fn load_data(file_path: &str) -> Result<(Array2<f64>, Array2<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut rdr = Reader::from_reader(file);

    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record);
    }

    if records.is_empty() {
        return Err("No data read from CSV file!".into());
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

    Ok((x, y))
}