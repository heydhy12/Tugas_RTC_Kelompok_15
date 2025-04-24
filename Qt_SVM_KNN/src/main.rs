mod data;
mod models;
mod visualization;

use crate::data::loader::load_data;
use crate::models::svm::SVMModel;
use linfa::dataset::Records;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Gunakan path absolut jika diperlukan
    let csv_path = "/home/heydhy12/Qt_SVM_KNN/csv/industry_maintenance.csv";
    println!("Membaca dataset dari: {}", csv_path);

    let dataset = load_data(csv_path).map_err(|e| {
        println!("Error loading data: {}", e);
        e
    })?;

    // Lanjutkan dengan proses training...
    Ok(())
}